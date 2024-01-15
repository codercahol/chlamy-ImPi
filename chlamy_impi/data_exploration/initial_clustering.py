from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from chlamy_impi.feature_extraction.prepare_feature_df import get_simple_y2_value_features, normalise_features, get_y2_features_continuous_light
from chlamy_impi.normalisation.y2_normalisation import get_normalised_y2_df


def main():
    df = get_normalised_y2_df()

    # First, as a sanity check, look at c=2 over all data. We expect two clusters, corresponding to HL and ML treatment
    sanity_check_hl_vs_ml(df)

    # Now create a dataset with 1 row per well, with features extracted from both light treatments
    feature_df = get_y2_features_continuous_light(df)
    feature_df = normalise_features(feature_df)
    feature_df = feature_df.dropna()

    # Get the positional indices of WT mutations in feature_df
    wt_pos_inds = get_wt_pos_inds(feature_df)

    feature_df = feature_df.iloc[:, 3:]

    # Make sure we can map back from positional index of feature_df to index of df
    positional_index_to_df_index = {i: index for i, index in enumerate(feature_df.index)}

    # Try clustering with different methods and parameters
    # For each one record the likelihood of each data point

    models = [GaussianMixture(n_components=1),
              GaussianMixture(n_components=2),
              GaussianMixture(n_components=4),
              GaussianMixture(n_components=8)]

    fig, axs = plt.subplots(2, 2, figsize=(12, 12), sharey=True, sharex=True)

    for i, model in enumerate(models):
        pred_labels = model.fit_predict(feature_df)
        scores = model.score_samples(feature_df)
        #plot_clusters_pca(feature_df, pred_labels, scores)

        ax = axs.ravel()[i]

        plot_clusters_with_outlying_genes(ax, df, feature_df, positional_index_to_df_index, pred_labels, scores,
                                          wt_pos_inds)

    plt.show()


def get_wt_pos_inds(feature_df):
    # Get the positional indices of WT rows in feature_df
    # This is somewhat tricky because feature_df has a different index to df
    def check_wt(feature_df_row):
        df_subset = df[(df["plate"] == feature_df_row["plate"]) & (df["i"] == feature_df_row["i"]) & (
                    df["j"] == feature_df_row["j"])]
        assert len(df_subset) <= 2
        if df_subset["mutant_ID"].iloc[0] == "WT":
            return True
        else:
            return False

    wt_df_inds = feature_df.apply(check_wt, axis=1)
    wt_pos_inds = [i for i, index in enumerate(feature_df.index) if wt_df_inds.loc[index]]
    assert len(wt_pos_inds) == 407  # = 384 + 8 * 3 with 1 missing?
    return wt_pos_inds


def plot_clusters_with_outlying_genes(ax, df, feature_df, positional_index_to_df_index, pred_labels, scores,
                                      wt_pos_inds):
    # Get the dataframe inds of the top-25 least likely points
    least_likely_inds = np.argsort(scores)[:10]
    scores = scores[least_likely_inds]
    least_likely_df_inds = [positional_index_to_df_index[i] for i in least_likely_inds]
    print(
        f"Least likely genes: {[(score, df.loc[i, 'mutated_genes']) for score, i in zip(scores, least_likely_df_inds)]}")

    # Plot all the wild type points, along with the least likely points, coloured by cluster
    pca = PCA(n_components=2)
    pca.fit(feature_df)
    pca_df = pca.transform(feature_df)
    ax.scatter(pca_df[wt_pos_inds, 0], pca_df[wt_pos_inds, 1], c=pred_labels[wt_pos_inds], marker="o", alpha=0.2,
               label="WT")
    ax.scatter(pca_df[least_likely_inds, 0], pca_df[least_likely_inds, 1], c=pred_labels[least_likely_inds], marker="x",
               alpha=1, label="Top-10 Outliers")

    # Annotate the gene name of the least likely points
    for i in least_likely_inds:
        ax.annotate(df.loc[i, "mutated_genes"].replace("&", "\&"), (pca_df[i, 0], pca_df[i, 1]))

    ax.legend()
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(f"GMM + PCA (explained var={sum(pca.explained_variance_ratio_):.2f}) with N_clusters={len(set(pred_labels))}")


def plot_clusters_pca(feature_df, pred_labels, pred_scores, dim=3):
    # Reduce dimensionality of data using PCA and then scatter plot
    pca = PCA(n_components=dim)
    pca.fit(feature_df)
    pca_df = pca.transform(feature_df)

    if dim==2:
        # Scatter plot of PCA data, coloured by likelihood
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        im = ax.scatter(pca_df[:, 0], pca_df[:, 1], c=pred_labels, marker="o", alpha=1)
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title(f"PCA (explained var={sum(pca.explained_variance_ratio_):.2f}) coloured by cluster")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.show()

    elif dim==3:
        # Scatter plot of PCA data, coloured by likelihood
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        im = ax.scatter(pca_df[:, 0], pca_df[:, 1], pca_df[:, 2], c=pred_labels, marker="o", alpha=1)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_zlabel("PCA 3")
        ax.set_title(f"PCA (explained var={sum(pca.explained_variance_ratio_):.2f}) coloured by cluster")
        plt.show()


def get_gene_to_indices(df, feature_df):
    gene_to_indices = defaultdict(list)
    for index in feature_df.index:
        mutated_genes = df.loc[index, "mutated_genes"].split(",")
        for gene in mutated_genes:
            gene_to_indices[gene].append(index)
    return gene_to_indices


def sanity_check_hl_vs_ml(df):
    # First, as a sanity check, look at c=2 over all data. We expect two clusters, corresponding to HL and ML treatment
    feature_df = get_simple_y2_value_features(df)
    feature_df = normalise_features(feature_df)
    feature_df = feature_df.dropna()

    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(feature_df)
    pred_labels = gmm.predict(feature_df)

    ml_mask = df["light_regime"] == "20h_ML"
    hl_mask = df["light_regime"] == "20h_HL"
    ml_df_index = df[ml_mask].index
    hl_df_index = df[hl_mask].index
    ml_inds = [i for i, index in enumerate(feature_df.index) if index in ml_df_index]
    hl_inds = [i for i, index in enumerate(feature_df.index) if index in hl_df_index]
    assert len(ml_inds) > 0
    assert len(hl_inds) > 0

    # Reduce dimensionality of data to 2D using PCA
    pca = PCA(n_components=2)
    pca.fit(feature_df)
    pca_df = pca.transform(feature_df)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)

    # Scatter plot of PCA data, coloured by cluster
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "Computer Modern"
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.set_cmap("tab20")
    ax.scatter(pca_df[ml_inds, 0], pca_df[ml_inds, 1], c=pred_labels[ml_inds], vmin=0, vmax=1, marker="x", alpha=1, label="ML")
    ax.scatter(pca_df[hl_inds, 0], pca_df[hl_inds, 1], c=pred_labels[hl_inds], marker="o", vmin=0, vmax=1, alpha=1, label="HL")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("PCA of features, coloured by GMM cluster")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()