import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import chi2
from scipy.stats import ttest_1samp
from scipy.stats import multivariate_normal
from scipy.special import comb
from scipy.special import erf
import scipy
from scipy.ndimage import gaussian_filter1d
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE
from umap import UMAP
from dtaidistance import dtw
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
import seaborn as sns

def get_clusters_whole_time_series(data, method, n_clusters=3, plot=False):
    data_cluster = data.copy()
    if method == 'pca':
        for light in data['light_regime'].unique():
            data_light = data[data['light_regime'] == light]
            data_light_y2 = data_light.filter(like='y2_').dropna(axis=1).values.astype(float)
            # Create a PCA model with 2 components: pca
            pca = PCA(n_components=2)

            kmeans = KMeans(n_clusters=n_clusters)

            # Make a pipeline chaining normalizer, pca and kmeans: pipeline
            pipeline = make_pipeline(StandardScaler(), pca, kmeans)
            pipeline.fit(data_light_y2)

            # Calculate the cluster labels: labels
            labels_pca = pipeline.predict(data_light_y2)

            # add the labels to the data
            data_cluster.loc[data_cluster['light_regime'] == light, 'cluster_' + method] = labels_pca
            
            if plot == True:
                # Plot the clusters
                plt.scatter(data_light_y2[:, 0], data_light_y2[:, 1], c=labels_pca, cmap='viridis')
                plt.xlabel('PCA 1')
                plt.ylabel('PCA 2')
                plt.title('PCA clustering')
                plt.show()

    elif method == 't-sne':
        for light in data['light_regime'].unique():
            data_light = data[data['light_regime'] == light]
            data_light_y2 = data_light.filter(like='y2_').dropna(axis=1).values.astype(float)
            
            tsne = TSNE(n_components=3)

            data_tsne = tsne.fit_transform(data_light_y2)

            kmeans = KMeans(n_clusters=n_clusters)

            pipeline = make_pipeline(kmeans)

            # Fit the pipeline to samples
            pipeline.fit(data_tsne)

            labels_tsne = pipeline.predict(data_tsne)

            data_cluster.loc[data_cluster['light_regime'] == light, 'cluster_' + method] = labels_tsne

            if plot == True:
                plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels_tsne, cmap='viridis')
                plt.xlabel('TSNE 1')
                plt.ylabel('TSNE 2')
                plt.title('TSNE clustering')
                plt.show()

    elif method == 'umap':
        for light in data['light_regime'].unique():
            data_light = data[data['light_regime'] == light]
            data_light_y2 = data_light.filter(like='y2_').dropna(axis=1).values.astype(float)
            
            umap = UMAP(n_components=2)

            data_umap = umap.fit_transform(data_light_y2)

            kmeans = KMeans(n_clusters=n_clusters)

            pipeline = make_pipeline(kmeans)

            # Fit the pipeline to samples
            pipeline.fit(data_umap)

            labels_umap = pipeline.predict(data_umap)

            data_cluster.loc[data_cluster['light_regime'] == light, 'cluster_' + method] = labels_umap

            if plot == True:
                # Plot the UMAP embedding
                plt.figure(figsize=(8, 6))
                plt.scatter(data_umap[:, 0], data_umap[:, 1], c=labels_umap, cmap='viridis')
                plt.xlabel('UMAP 1')
                plt.ylabel('UMAP 2')
                plt.title('UMAP Embedding Clustering')
                plt.show()
    
    return data_cluster

def plot_clusters(data, method, n=50):
    n_clusters = len(data['cluster_' + method].unique())
    fig, axs = plt.subplots(nrows=(n_clusters - 1)//3 + 1, ncols=3, figsize=(15, 5*(n_clusters - 1)//3 + 1))
    for i, cluster in enumerate(data['cluster_' + method].unique()):
        data_cluster = data[data['cluster_' + method] == cluster]
        data_cluster_y2 = data_cluster.filter(like='y2_').dropna(axis=1).values.astype(float)
        if n_clusters <= 3:
            axs[i].plot(data_cluster_y2[:n].T, alpha=0.5)
            axs[i].set_title('Cluster ' + str(i))
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel('Y')
        else :
            axs[i//3, i%3].plot(data_cluster_y2[:n].T)
            axs[i//3, i%3].set_title('Cluster ' + str(cluster))
            axs[i//3, i%3].set_xlabel('Time')
    plt.tight_layout()
    plt.show()

def grad_distance(s1, s2):
    s1_grad = np.gradient(s1)
    s2_grad = np.gradient(s2)
    return np.linalg.norm(s1_grad - s2_grad)**2

def dtw_distance(s1, s2):
    return dtw.distance(s1, s2)

def dtw_grad_distance(s1, s2):
    s1_grad = np.gradient(s1)
    s2_grad = np.gradient(s2)
    return dtw.distance(s1_grad, s2_grad)

def wasserstein_distance(s1, s2):
    # Create histograms
    hist1, bin_edges1 = np.histogram(s1, bins=50, density=True)
    hist2, bin_edges2 = np.histogram(s2, bins=50, density=True)

    # Calculate bin centers
    bin_centers1 = (bin_edges1[:-1] + bin_edges1[1:]) / 2
    bin_centers2 = (bin_edges2[:-1] + bin_edges2[1:]) / 2

    # Calculate the Wasserstein distance
    return scipy.stats.wasserstein_distance(bin_centers1, bin_centers2, hist1, hist2)

def euclidian_distance(s1, s2):
    return np.linalg.norm(s1 - s2)**2

def dim_reduc(data, method, light_regime, metric=euclidian_distance, n_components=2, n_samples=1000, plot=True):
    data_light = data[data['light_regime'] == light_regime].iloc[:n_samples]
    features = data[data['light_regime'] == light_regime].iloc[:n_samples].filter(like='y2_').dropna(axis=1).values

    if method == 'pca':
        pca = PCA(n_components=15)
        features_red = pca.fit_transform(features)
        # Plot the explained variance ratio
        plt.bar(range(15), pca.explained_variance_ratio_)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance Ratio of PCA')
        plt.show()
        pca = PCA(n_components=2)
        features_red = pca.fit_transform(features)
    elif method == 't-sne':
        tsne = TSNE(n_components=n_components)
        features_red = tsne.fit_transform(features)
    elif method == 'umap':
        umap = UMAP(n_components=n_components)
        features_red = umap.fit_transform(features)
    elif method == 'kpca':
        # # Compute the kernel matrix
        # kernel_matrix = pairwise_kernels(features_HL, metric=metric)

        # # Perform Kernel PCA
        # kpca = KernelPCA(n_components=2, kernel='precomputed')
        # features_red = kpca.fit_transform(kernel_matrix)

        kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=15)
        features_red = kpca.fit_transform(features)

    elif method == 'lle':
        lle = LocallyLinearEmbedding(n_components=2)
        features_red = lle.fit_transform(features)
    elif method == 'isomap':
        isomap = Isomap(n_components=2)
        features_red = isomap.fit_transform(features)
    elif method == 'mds':
        # subsample each time-series of features_HL
        subsampling_rate = 4
        features_subsampled = features[:, ::subsampling_rate]

        distance_matrix = np.zeros((features_subsampled.shape[0], features_subsampled.shape[0]))
        for i in range(features_subsampled.shape[0]):
            for j in range(features_subsampled.shape[0]):
                distance_matrix[i, j] = metric(features_subsampled[i], features_subsampled[j])
        
        mds = MDS(n_components=2, dissimilarity='precomputed')
        features_red = mds.fit_transform(distance_matrix)
    elif method == 'spectral':
        se = SpectralEmbedding(n_components=2, random_state=42)
        features_red = se.fit_transform(features)
    elif method == 'gaussian_random':
        random_proj = GaussianRandomProjection(n_components=n_components, random_state=42)
        features_red = random_proj.fit_transform(features)
    else:
        raise ValueError('Method not implemented')
    
    if plot == True:
        # Plot the embedding and color the points according if they are WT, end_down or beg_down outliers
        plt.figure(figsize=(8, 6))
        plt.scatter(features_red[:, 0], features_red[:, 1], c='blue', s=10)
        # plot the right color the outliers
        plt.scatter(features_red[data_light['mutant_ID'] == 'WT', 0], features_red[data_light['mutant_ID'] == 'WT', 1], c='green', s=10, label='WT')
        plt.scatter(features_red[data_light['outlier_divergent_end_down_y2'] == True, 0], features_red[data_light['outlier_divergent_end_down_y2'] == True, 1], c='red', s=10, label='end_down')
        plt.scatter(features_red[data_light['outlier_divergent_beg_down_y2'] == True, 0], features_red[data_light['outlier_divergent_beg_down_y2'] == True, 1], c='purple', s=10, label='beg_down')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(method + ' Embedding of ' + light_regime + ' time-series')
        plt.legend()
        plt.show()

    return features_red

def get_clusters(data, features_red, n_clusters=3):
    data_HL = data[data['light_regime'] == '20h_HL']
    # Perform clustering on the red components
    kmeans = KMeans(n_clusters=5, random_state=42)
    data_HL['Cluster'] = kmeans.fit_predict(features_red)

    # plot the clusters
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data_HL, x=features_red[:, 0], y=features_red[:, 1], hue='Cluster', palette='viridis', s=10)
    plt.title('T-SNE Clustering of 20h_HL')
    plt.show()

    # plot the y2 from each clusters
    for cluster in data_HL['Cluster'].unique():
        data_y2 = data_HL[(data_HL['Cluster'] == cluster) & (data_HL['light_regime'] == '20h_HL')].filter(like='y2_').dropna(axis=1).values
        plt.figure(figsize=(8, 6))
        for i in range(data_y2.shape[0]):
            plt.plot(data_y2[i], c='blue', alpha=0.5)
        plt.title(f'Cluster {cluster} - y2')
        plt.xlabel('Time')
        plt.ylabel('Y2')

def do_pca_on_mutants(mutant_data, features, n_components=2, plot=True):
    # Drop rows with NaN values for PCA
    # X = mutant_data.drop(['mutant_ID', 'mutated_genes', 'plate', 'well_id', 'GO', 'aggregated_GO'], axis=1)
    X = mutant_data[features]
    X_nonan = X.dropna()

    # Step 1: Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_nonan)

    # Step 2: Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    loadings = pca.components_

    # Create a DataFrame for better readability
    loadings_df = pd.DataFrame(loadings, columns=[X.columns[i] for i in range(X.shape[1])])
    loadings_df.index = [f'PC {i+1}' for i in range(loadings.shape[0])]

    # Step 3: Create a DataFrame with the PCA results
    pca_df = pd.DataFrame(data=X_pca, columns=['PC'+ str(i) for i in range(1, n_components + 1)], index=X_nonan.index)

    # Step 4: Merge PCA results back to the original DataFrame
    mutant_data_with_pca = mutant_data.join(pca_df, how='left')

    return mutant_data_with_pca, loadings_df

# Varimax function
def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for i in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        u, s, vh = np.linalg.svd(np.dot(Phi.T, np.asarray(Lambda)**3 - (gamma/p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda))))))
        R = np.dot(u, vh)
        d = np.sum(s)
        if d_old != 0 and d/d_old < 1 + tol:
            break
    return np.dot(Phi, R)

def do_pca_on_genes(gene_data, n_components=2, plot=True):
    # Drop rows with NaN values for PCA
    X = gene_data.drop(['mutated_genes', 'GO', 'aggregated_GO'], axis=1)
    X_nonan = X.dropna()

    # Step 1: Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_nonan)

    # Step 2: Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    loadings = pca.components_

    # Create a DataFrame for better readability
    loadings_df = pd.DataFrame(loadings, columns=[X.columns[i] for i in range(X.shape[1])])
    loadings_df.index = [f'PC {i+1}' for i in range(loadings.shape[0])]

    # Apply Varimax rotation to the loadings
    rotated_loadings = varimax(loadings)

    # Convert to DataFrame for easier interpretation
    rotated_loadings_df = pd.DataFrame(rotated_loadings, index=[f'PC {i+1}' for i in range(loadings.shape[0])], columns=[X.columns[i] for i in range(X.shape[1])])

    # Step 3: Create a DataFrame with the PCA results
    pca_df = pd.DataFrame(data=X_pca, columns=['PC'+ str(i) for i in range(1, n_components + 1)], index=X_nonan.index)

    # Step 4: Merge PCA results back to the original DataFrame
    gene_data_with_pca = gene_data.join(pca_df, how='left')

    return gene_data_with_pca, loadings_df, rotated_loadings_df

def check_explained_variance(gene_data, features):
    # Drop rows with NaN values for PCA
    X = gene_data[features]
    X_nonan = X.dropna()

    # Step 1: Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_nonan)

    # Step 2: Perform PCA
    pca = PCA(n_components=X_scaled.shape[1])
    X_pca = pca.fit_transform(X_scaled)
    # show the explained variance
    arr = np.array([0] + list(pca.explained_variance_ratio_))
    plt.plot(np.cumsum(arr))
    plt.title('Explained variance')

def plot_pca_and_region(gene_data_with_pca, region = [0, 1, 0, 1], components = [1, 2], overlay=None):
    # get a mutant with PCA in a certain region
    region_genes = gene_data_with_pca[(gene_data_with_pca['PC' + str(components[0])] > region[0]) & (gene_data_with_pca['PC' + str(components[0])] < region[1]) & (gene_data_with_pca['PC' + str(components[1])] > region[2]) & (gene_data_with_pca['PC' + str(components[1])] < region[3])]
    plt.figure(figsize=(10, 6))
    plt.scatter(gene_data_with_pca['PC' + str(components[0])], gene_data_with_pca['PC' + str(components[1])], s=10, alpha=0.5, c='blue')
    plt.scatter(gene_data_with_pca[gene_data_with_pca['mutated_genes'] == '']['PC' + str(components[0])], gene_data_with_pca[gene_data_with_pca['mutated_genes'] == '']['PC' + str(components[1])], color='green', s=10, alpha=1)
    if overlay is not None:
        plt.scatter(overlay['PC' + str(components[0])], overlay['PC' + str(components[1])], color='red', s=10, alpha=0.5)
    plt.scatter(region_genes['PC' + str(components[0])], region_genes['PC' + str(components[1])], color='red', s=10, alpha=0.5)
    # plot the region
    plt.plot([region[0], region[1], region[1], region[0], region[0]], [region[2], region[2], region[3], region[3], region[2]], color='red')
    plt.xlabel('Principal Component ' + str(components[0]))
    plt.ylabel('Principal Component ' + str(components[1]))
    plt.title('PCA of Mutant Data')
    plt.show()
    return region_genes
        
    