from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics.pairwise import euclidean_distances

from chlamy_impi.normalisation.y2_normalisation import get_normalised_y2_df


def main():
    pd.options.display.max_columns = 100
    df = get_normalised_y2_df()
    df = df[df["light_regime"] == "20h_HL"]
    print(df.head())

    y2_series_df = create_y2_df(df)

    # Compute euclidean distance between each one (assuming they're all the same length)
    distances = euclidean_distances(y2_series_df)

    # Plot the similarity matrix
    plt.imshow(distances)
    plt.show()

    # Now let's look at genetically identical clusters. Get sets of indices corresponding to rows with identical mutants from df
    genes_to_indices = construct_gene_to_df_inds_map(df, y2_series_df)

    # Dataframe index to similarity matrix index
    df_index_to_matrix_index = {index: i for i, index in enumerate(y2_series_df.index)}

    # Now for each gene, compute the average similarity between all wells which have that gene knocked out
    # Multiple knockouts are a confounder here which isn't handled properly yet
    gene_to_avg_similarity, wt_similarity = compute_average_intragene_similarity(df_index_to_matrix_index, distances,
                                                                                 genes_to_indices)

    # Print out the top 10 and bottom 10 scoring genes
    sorted_genes = sorted(gene_to_avg_similarity.items(), key=lambda x: x[1][1])
    sorted_gene_names = [gene.replace("&", "\&") for gene, similarity in sorted_genes]

    plot_intra_gene_similarity(gene_to_avg_similarity, sorted_gene_names, wt_similarity)

    # Based on the previous plots, let's used 0.5 as a threshold for self-similarity
    good_genes = [gene for gene, similarity in gene_to_avg_similarity.items() if similarity[1] < 0.5]

    # Okay, now let's go through these good genes and see how similar they are to wild type
    # Let's get all WT indices in the similarity matrix
    gene_to_wt_similarity = compute_gene_to_wt_similarity(df_index_to_matrix_index, distances, genes_to_indices,
                                                          good_genes)

    plot_gene_to_wt_similarity_histogram(gene_to_wt_similarity)

    # Extract genes with distance score of greater than 0.5
    candidate_genes = [gene for gene, similarity in gene_to_wt_similarity.items() if similarity > 0.5]
    candidate_genes = sorted(candidate_genes, key=lambda x: gene_to_wt_similarity[x], reverse=True)

    print(f"Number of candidate genes: {len(candidate_genes)}")
    print(f"Top 16 candidate genes: ")
    for gene in candidate_genes[:16]:
        print(f"{gene}: {gene_to_wt_similarity[gene]}")

    plot_top_outlier_time_series(candidate_genes, gene_to_wt_similarity, genes_to_indices, y2_series_df)


def plot_top_outlier_time_series(candidate_genes, gene_to_wt_similarity, genes_to_indices, y2_series_df):
    # Plot the Y(II) time series of the top-16 candidate genes!
    fig, axs = plt.subplots(4, 4, figsize=(10, 5), sharex=True, sharey=True)
    for i, gene in enumerate(candidate_genes[:16]):
        ax = axs.ravel()[i]
        ax.set_ylim(0, 1)
        for index in genes_to_indices[gene]:
            ax.plot(y2_series_df.loc[index, ["fv_fm"] + [f"y2_{i}" for i in range(1, 42)]], color="red", alpha=1)
        for index in genes_to_indices[""]:
            ax.plot(y2_series_df.loc[index, ["fv_fm"] + [f"y2_{i}" for i in range(1, 42)]], color="black", alpha=0.1)
        ax.set_title(f"{gene} (score={gene_to_wt_similarity[gene]:.2f})")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Only show a couple of xticks
    axs[0, 0].set_xticks([0, 20, 40])
    axs[0, 0].set_xticklabels([0, 20, 40])
    axs[0, 0].set_ylabel("Y(II)")
    axs[3, 0].set_xlabel("Time point")
    plt.show()


def plot_gene_to_wt_similarity_histogram(gene_to_wt_similarity):
    # Plot histogram of WT similarity scores
    plt.clf()
    plt.hist(gene_to_wt_similarity.values(), bins=100)
    plt.xlabel("Average Euclidean distance between genes and WT")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()


def compute_gene_to_wt_similarity(df_index_to_matrix_index, distances, genes_to_indices, good_genes):
    """Compute the average similarity between each gene and wild type, by averaging all pairwise distances"""
    wt_indices = [df_index_to_matrix_index[i] for i in genes_to_indices[""]]
    assert len(wt_indices) >= 400
    gene_to_wt_similarity = {}
    for gene in good_genes:
        gene_indices = [df_index_to_matrix_index[i] for i in genes_to_indices[gene]]
        gene_to_wt_similarity[gene] = np.mean([distances[i, j] for i in gene_indices for j in wt_indices])
    return gene_to_wt_similarity


def plot_intra_gene_similarity(gene_to_avg_similarity, sorted_gene_names, wt_similarity):
    """Plot the similarities within each gene knockout by grouping replicates together"""
    xs = [similarity[0] for gene, similarity in gene_to_avg_similarity.items()]
    ys = [similarity[1] for gene, similarity in gene_to_avg_similarity.items()]
    names = [gene for gene, similarity in gene_to_avg_similarity.items()]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(xs, ys, marker="x")

    # Annotate names of top 10 and bottom 10
    for i, name in enumerate(names):
        if name in sorted_gene_names[-100:]:
            ax.annotate(name, (xs[i], ys[i]))
    ax.axhline(wt_similarity, color="red", linestyle="--")
    ax.set_xlabel("Number of repeats")
    ax.set_ylabel("Average Euclidean distance between repeats")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()

    # Plot histogram of similarity scores
    plt.clf()
    plt.hist(ys, bins=100)
    plt.xlabel("Average Euclidean distance between repeats")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()


def compute_average_intragene_similarity(df_index_to_matrix_index, distances, genes_to_indices):
    """Compute the average similarity between all replicates of a particular gene knockout

    TODO: How should multiple knockouts be handled?
    """
    gene_to_avg_similarity = {}
    for gene, indices in genes_to_indices.items():
        if len(indices) == 1:
            continue

        if gene == "":
            gene = "WT"

        matrix_inds = [df_index_to_matrix_index[i] for i in indices]
        matrix_ind_pairs = [(matrix_inds[i], matrix_inds[j]) for i, j in combinations(range(len(matrix_inds)), 2)]
        gene_to_avg_similarity[gene] = (len(indices), np.mean([distances[i, j] for i, j in matrix_ind_pairs]))
    for gene, similarity in gene_to_avg_similarity.items():
        print(f"{gene}: {similarity}")

    wt_similarity = gene_to_avg_similarity["WT"][1]
    del (gene_to_avg_similarity["WT"])

    return gene_to_avg_similarity, wt_similarity


def construct_gene_to_df_inds_map(df, y2_series_df):
    """Construct a mapping from gene name to a list of dataframe indices corresponding to wells with that gene"""
    genes_to_indices = defaultdict(list)
    for index in y2_series_df.index:
        mutated_genes = df.loc[index, "mutated_genes"].split(",")
        for gene in mutated_genes:
            genes_to_indices[gene].append(index)
    return genes_to_indices


def create_y2_df(df):
    """Create a dataframe with one row per well, and one column for each time point in Y(II) time series"""
    y2_series_df = df[["fv_fm"] + [f"y2_{i}" for i in range(1, 42)]]  # 42 is max for continuous light

    # Drop rows where all values are nan to eliminate blank wells
    y2_series_df = y2_series_df.dropna(how="all")
    assert y2_series_df.isna().sum().sum() == 0, "Some NaNs still present"

    # Smooth each time series using a rolling average (excluding the first and last column)
    y2_series_df[[f"y2_{i}" for i in range(1, 41)]] = \
        y2_series_df[[f"y2_{i}" for i in range(1, 41)]].rolling(5, min_periods=1, axis=1).mean()

    return y2_series_df


if __name__ == "__main__":
    main()
