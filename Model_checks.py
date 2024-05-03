import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import normaltest
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency
from scipy.stats import levene
import seaborn as sns
from scipy.stats import multivariate_normal
from pingouin import multivariate_normality
from scipy.stats import kstest
from scipy.spatial.distance import pdist, squareform
import scipy

def plot_y2_normalized_distribution(data) :
    data_y2 = data.filter(like='y2_').dropna(axis=1).values

    # Check if the distribution is Gaussian for each time point and plot the associated gaussian distribution
    fig, axes = plt.subplots(8, len(data_y2[0])//8, figsize=(15, 20))  # Create a 8x5 grid of subplots

    for i, ax in enumerate(axes.flatten()):
        ax.hist(data_y2[:, i], bins=20, density=True, alpha=0.6, color='b')
        x = np.linspace(data_y2[:, i].min(), data_y2[:, i].max(), 100)
        y = multivariate_normal.pdf(x, mean=np.mean(data_y2[:, i]), cov=np.cov(data_y2[:, i]))
        ax.plot(x, y, 'r--')
        ax.set_title('Time point ' + str(i))

    plt.tight_layout()
    plt.show()

def plot_ynpq_normalized_distribution(data) :
    data_ynpq = data.filter(like='ynpq_').dropna(axis=1).values

    # Check if the distribution is Gaussian for each time point and plot the associated gaussian distribution
    fig, axes = plt.subplots(8, len(data_ynpq[0])//8, figsize=(15, 20))  # Create a 8x5 grid of subplots

    for i, ax in enumerate(axes.flatten()):
        ax.hist(data_ynpq[:, i], bins=20, density=True, alpha=0.6, color='b')
        x = np.linspace(data_ynpq[:, i].min(), data_ynpq[:, i].max(), 100)
        y = multivariate_normal.pdf(x, mean=np.mean(data_ynpq[:, i]), cov=np.cov(data_ynpq[:, i]))
        ax.plot(x, y, 'r--')
        ax.set_title('Time point ' + str(i))

    plt.tight_layout()
    plt.show()

def plot_y2_distribution(data) :
    data_y2 = data.filter(like='y2_').dropna(axis=1).values

    # Check if the distribution is Gaussian for each time point and plot the associated gaussian distribution
    fig, axes = plt.subplots(8, len(data_y2[0])//8, figsize=(15, 20))  # Create a 8x5 grid of subplots

    for i, ax in enumerate(axes.flatten()):
        ax.hist(data_y2[:, i]/(np.sum(data_y2[:, i])), bins=20)
        ax.set_title('Time point ' + str(i))

    plt.tight_layout()
    plt.show()

def normal_test(data) :
    data_y2 = data.filter(like='y2_').dropna(axis=1)
    # make the data into a probability distribution by making it sum to 1

    # Perform the normality test for each time point
    normality_test = pd.DataFrame(index=['statistic', 'p-value'], columns=data_y2.columns)

    for column in data_y2.columns:
        ks_test = kstest(data_y2[column], 'norm', args=(np.mean(data_y2[column]), np.std(data_y2[column])))
        statistic, p_value = ks_test.statistic, ks_test.pvalue
        normality_test[column] = [statistic, p_value]

    # print('Are the time points normally distributed?', 'Yes' if (normality_test.loc['p-value'] > 0.05).all() else 'No')

    return normality_test

def ks_test(data) :
    # Generate some sample data (replace this with your data)
    np.random.seed(0)
    samples = np.random.normal(loc=0, scale=1, size=100)

    # Step 1: Estimate Empirical Probability Distribution (ECDF)
    def empirical_distribution(samples):
        sorted_samples = np.sort(samples)
        n = len(sorted_samples)
        ecdf = np.arange(1, n + 1) / n
        return sorted_samples, ecdf

    sorted_samples, ecdf = empirical_distribution(samples)

    # Step 2: Fit a Normal Distribution to the samples
    mu, sigma = np.mean(samples), np.std(samples)
    norm_dist = stats.norm(mu, sigma)

    # Step 3: Perform KS Test
    ks_statistic, ks_p_value = stats.kstest(sorted_samples, norm_dist.cdf)

    # Print results
    print("KS Statistic:", ks_statistic)
    print("KS Test p-value:", ks_p_value)

    # Plot ECDF and Normal CDF for visualization (optional)

    plt.figure(figsize=(8, 5))
    plt.plot(sorted_samples, ecdf, label='Empirical CDF')
    plt.plot(sorted_samples, norm_dist.cdf(sorted_samples), label='Normal CDF')
    plt.xlabel('Samples')
    plt.ylabel('Cumulative Probability')
    plt.title('Empirical CDF vs. Normal CDF')
    plt.legend()
    plt.grid(True)
    plt.show()


def independence_test(data) :
    data_y2 = data.filter(like='y2_').dropna(axis=1)

    # Perform the independence test for each pair of time points
    independence_test = pd.DataFrame(index=data_y2.columns, columns=data_y2.columns)

    for column1 in data_y2.columns:
        for column2 in data_y2.columns:
            if column1 != column2:
                statistic, p_value = pearsonr(data_y2[column1], data_y2[column2])
                independence_test.loc[column1, column2] = statistic

    # chi2_value, p, _, _ = chi2_contingency(data_y2.corr())
    # print('Chi-squared statistic:', chi2_value)
    # print('P-value:', p)
    # print('Are the time points independent?', 'Yes' if p > 0.05 else 'No')

    return independence_test

def plot_corr(data):
    data_y2 = data.filter(like='y2_').dropna(axis=1)
    plt.figure(figsize=(12, 10))
    sns.heatmap(data_y2.corr(), cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Y(II) Measurements')
    plt.show()

def plot_cov(data) :
    data_y2 = data.filter(like='y2_').dropna(axis=1)
    plt.figure(figsize=(12, 10))
    sns.heatmap(np.cov(data_y2, rowvar=False), cmap='coolwarm', center=0)
    plt.title('Covariance Matrix of Y(II) Measurements')
    plt.show()

def same_variance_test(data) :
    # Checks if the variance for each time-point is the same (ie that the cov matrix of espilon is an homothetie of the identity matrix)
    # Perform the Levene test for the y2_ columns
    data_y2 = data.filter(like='y2_').dropna(axis=1)
    statistic, p_value = levene(*[data_y2[column] for column in data_y2])
    print('Levene test statistic:', statistic)
    print('P-value:', p_value)
    print('Are the variances equal?', 'Yes' if p_value > 0.05 else 'No')

def test_multivariate_gaussian(data) :
    # Perform the Mardia test for multivariate normality
    data_y2 = data.filter(like='y2_').dropna(axis=1)
    statistic, p_value, res  = multivariate_normality(data_y2, alpha=0.05)
    print('Mardia test statistic:', statistic)
    print('P-value:', p_value)
    print('Is the data multivariate Gaussian?', 'Yes' if res else 'No')

def plot_mean_var_per_light_regime(data) :
    mean_vars = []
    for light in data['light_regime'].unique():
        data_light = data[data['light_regime'] == light]
        data_light_y2 = data_light.filter(like='y2_').dropna(axis=1).values
        cov_matrix = np.cov(data_light_y2, rowvar=False)
        mean_var = np.mean(np.diag(cov_matrix))
        mean_vars.append(mean_var)
    plt.bar(data['light_regime'].unique(), mean_vars)

def plot_corr_per_light_regime(data) :
    for light in data['light_regime'].unique():
        data_light = data[data['light_regime'] == light]
        data_light_y2 = data_light.filter(like='y2_').dropna(axis=1).values
        plt.figure(figsize=(10, 8))
        sns.heatmap(np.corrcoef(data_light_y2, rowvar=False), cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Y(II) Measurements in ' + light)
    plt.tight_layout()
    plt.show()

def plot_distance_matrix(data) :
    distance_matrixes = []
    for light in data['light_regime'].unique():
        data_light = data[data['light_regime'] == light]
        data_light_y2 = data_light.filter(like='y2_').dropna(axis=1)
        for i in range(len(data_light_y2)):
            distances = pdist(data_light_y2.iloc[i].values.reshape(-1, 1), metric='euclidean')
            distance_matrix = squareform(distances)
            distance_matrixes.append(distance_matrix)
        mean_distance = np.mean(distance_matrixes, axis=0)
        plt.figure(figsize=(10, 8))
        sns.heatmap(mean_distance, cmap='coolwarm', center=0)
        plt.title('Distance Matrix of Y(II) Measurements in ' + light)
    plt.tight_layout()
    plt.show()

def simulate_gaussian_vectors(n, mean, cov):
    # Generate n samples of Gaussian vectors with dimension d and covariance matrix cov
    samples = np.random.multivariate_normal(mean=mean, cov=cov, size=n)
    
    return samples

def plot_reducted_gaussian_vectors(data):
    for light in data['light_regime'].unique():
        data_light = data[data['light_regime'] == light]
        data_light_y2 = data_light.filter(like='y2_').dropna(axis=1)
        n = len(data_light_y2.iloc[0])      
        cov_matrix = np.cov(data_light_y2, rowvar=False)
        sqrt_inv = scipy.linalg.sqrtm(np.linalg.inv(cov_matrix))
        data_light_y2_reducted = np.dot(data_light_y2, sqrt_inv)

    # plot the vectors and the normalized vectors in two fig next to each others for each light
        time_points = np.arange(0, n)
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for i in range(300):
            axes[0].plot(time_points, data_light_y2.iloc[i])
            axes[0].set_title('Original (normalized) Y(II) responses in ' + light)
            # set x ticks
            axes[1].plot(time_points, data_light_y2_reducted[i])
            axes[1].set_title('Reduced (normalized) Y(II) responses in ' + light)
        plt.tight_layout()
        plt.show()

def plot_diagonal_reducted_gaussian_vectors(data):
    for light in data['light_regime'].unique():
        data_light = data[data['light_regime'] == light]
        data_light_y2 = data_light.filter(like='y2_').dropna(axis=1)
        n = len(data_light_y2.iloc[0])      
        cov_matrix = np.cov(data_light_y2, rowvar=False)
        sqrt_inv = scipy.linalg.sqrtm(np.linalg.inv(np.diag(np.diag(cov_matrix))))
        data_light_y2_reducted = np.dot(data_light_y2, sqrt_inv)

    # plot the vectors and the normalized vectors in two fig next to each others for each light
        time_points = np.arange(0, n)
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for i in range(300):
            axes[0].plot(time_points, data_light_y2.iloc[i])
            axes[0].set_title('Original (normalized) Y(II) responses in ' + light)
            # set x ticks
            axes[1].plot(time_points, data_light_y2_reducted[i])
            axes[1].set_title('Reduced (normalized) Y(II) responses in ' + light)
        plt.tight_layout()
        plt.show()
