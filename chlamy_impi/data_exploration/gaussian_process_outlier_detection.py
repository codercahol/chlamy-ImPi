"""

NOTE: This code is a prototype of a Gaussian Process + Hypothesis Testing approach to identify mutants which are
significantly different to WT. It is designed to illustrate and test out the approach, but the results shouldn't
yet be taken seriously. The most important next step is that we need to group mutants according to the gene which is
knocked out, and then run a hypothesis test such that only mutants which are consistent enough between replicates, as
well as significantly different to WT, are highlighted as outliers. However, so far it looks promising, and I like the
approach.

"""

import numpy as np
from scipy.stats import chi2, multivariate_normal
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF

from chlamy_impi.data_exploration.time_series_similarity import create_y2_df, construct_gene_to_df_inds_map
from chlamy_impi.normalisation.y2_normalisation import get_normalised_y2_df


def main():
    df = get_normalised_y2_df()
    df = df[df["light_regime"] == "20h_HL"]  # At present, just run this for HL data

    mutant_series, wt_indices, wt_series = separate_mutants_and_wt(df)

    # I need to reduce the number of samples considered here, because the GP is too slow - random sample a subset
    wt_series_subset = wt_series.sample(50)

    # Cut the first and last columns, the discontinuity is difficult to model with a GP with a simple kernel function
    wt_series_subset = wt_series_subset.iloc[:, 1:-1]
    mutant_series = mutant_series.iloc[:, 1:-1]

    # Apply a rolling average to the time series to smooth out noise (drop the nans)
    wt_series_subset = wt_series_subset.rolling(window=9, axis=1, min_periods=9, center=True).mean().dropna(axis=1)
    mutant_series = mutant_series.rolling(window=9, axis=1, min_periods=9, center=True).mean().dropna(axis=1)

    # Prepare data and label matrix of WT data, which we will fit GP to
    X = np.tile(np.linspace(0, 1, len(wt_series_subset.columns)), len(wt_series_subset)).reshape(-1, 1)
    y = wt_series_subset.values.ravel()

    plot_wt_variance(wt_series)

    gp = fit_gp_to_wt(X, wt_series, y)

    plot_trained_gp(X, gp, y)

    # Get mu and K for the GP so we can assess likelihood of other time series defined over the same time points
    mu, K = gp.predict(np.linspace(0, 1, len(wt_series_subset.columns)).reshape(-1, 1), return_cov=True)

    plot_mu_and_K(K, mu)

    log_likelihoods, mutant_inds, null_hypothesis_results = run_all_null_hypothesis_tests(K, mu, mutant_series)

    plot_least_and_most_likely_mutants(df, log_likelihoods, mutant_inds, mutant_series, null_hypothesis_results,
                                       wt_indices, wt_series)

    plot_all_mutants(df, log_likelihoods, mutant_inds, mutant_series, null_hypothesis_results,
                                       wt_indices, wt_series)

    # TODO: at present we aren't filtering out mutants which have inconsistent replicates
    # One way to extend the current approach to this is as follows:
        # Find the mean value of all replicates at each time point
        # Fit a GP to these values with the "alpha" parameter set to the magnitude of the noise learned by the WhiteKernel
        # during the MLE fit to WT data (i.e. our best estimate of the uncertainty between genetically identical replicates)
        # Then, we can perform N hypothesis tests (with Bonferroni correction?) to test whether any replicate is significantly
        # different to the mean? - Or maybe we do a pairwise comparison where for each pair, we fit a GP to one and test
        # if the other is significantly different (because the means would be affected by outliers). For multiple replicates
        # then a multi-way significance test such as ANOVA could be appropriate.


def plot_least_and_most_likely_mutants(df, log_likelihoods, mutant_inds, mutant_series, null_hypothesis_results,
                                       wt_indices, wt_series):
    """Plot the original time series for the WT, and a selection of mutants with the highest and lowest likelihoods
    """

    # First sort by log likelihood
    sorted_inds = np.argsort(log_likelihoods)
    sorted_inds = sorted_inds[::-1]  # Reverse so that highest likelihood is first
    N = 100
    top_inds = sorted_inds[:N]
    bottom_inds = sorted_inds[-N:]

    # Plot all the WT time series
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for i in wt_indices:
        ax.plot(wt_series.loc[i].iloc[1:-1], color="black", alpha=0.1)
    for i in top_inds:
        ax.plot(mutant_series.loc[mutant_inds[i]], color="red", alpha=1)
    for i in bottom_inds:
        ax.plot(mutant_series.loc[mutant_inds[i]], color="blue", alpha=1)

    # Add manual labels for legend
    ax.plot([], [], color="black", alpha=0.1, label="WT")
    ax.plot([], [], color="red", alpha=1, label=f"{N} most likely mutants")
    ax.plot([], [], color="blue", alpha=1, label=f"{N} least likely mutants")
    ax.legend()
    plt.show()

    print(f"Top 10 mutants: {[mutant_series.loc[mutant_inds[i]].name for i in top_inds]}")
    print(f"Null hypothesis for top-{N}: {[null_hypothesis_results[i] for i in top_inds]}")
    print(f"Log likelihoods for top-{N}: {[log_likelihoods[i] for i in top_inds]}")
    print(f"Genes for top-{N}: {[df.loc[mutant_series.loc[mutant_inds[i]].name, 'mutated_genes'] for i in top_inds]}")
    print(f"Bottom 10 mutants: {[mutant_series.loc[mutant_inds[i]].name for i in bottom_inds]}")
    print(f"Null hypothesis for bottom-{N}: {[null_hypothesis_results[i] for i in bottom_inds]}")
    print(f"Log likelihoods for bottom-{N}: {[log_likelihoods[i] for i in bottom_inds]}")
    print(
        f"Genes for bottom-{N}: {[df.loc[mutant_series.loc[mutant_inds[i]].name, 'mutated_genes'] for i in bottom_inds]}")


def plot_all_mutants(df, log_likelihoods, mutant_inds, mutant_series, null_hypothesis_results,
                                       wt_indices, wt_series):
    """Plot the original time series for the WT, and all mutants
    """
    top_inds = np.nonzero(null_hypothesis_results)[0]
    significant_difference_inds = np.nonzero(np.logical_not(null_hypothesis_results))[0]

    # Plot all the WT time series
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for i in wt_indices:
        ax.plot(wt_series.loc[i].iloc[1:-1], color="black", alpha=0.1)
    for i in top_inds:
        ax.plot(mutant_series.loc[mutant_inds[i]], color="red", alpha=0.3)
    for i in significant_difference_inds:
        ax.plot(mutant_series.loc[mutant_inds[i]], color="blue", alpha=0.3)

    # Add manual labels for legend
    ax.plot([], [], color="black", alpha=0.1, label="WT")
    ax.plot([], [], color="red", alpha=1, label=f"No significant difference to WT")
    ax.plot([], [], color="blue", alpha=1, label=f"Significant difference to WT")
    ax.legend()
    plt.show()

def run_all_null_hypothesis_tests(K, mu, mutant_series):
    """For each mutant, compute the likelihood of that time series according to the GP, and run a hypothesis test

    TODO: Is it statistically valid to perform repeated hypothesis tests? I think that because this is a screen, the
    type 1 errors (false positives) aren't a problem. For example, using this approach, if every mutant was actually
    just a WT replicate, we would erroneously find that 5% of our screened mutants are significantly different to WT.

    """
    null_hypothesis_results = []
    log_likelihoods = []
    mutant_inds = []

    for i, row in mutant_series.iterrows():
        row = row.values
        null_hypothesis_results.append(null_hypothesis_test(mu, K, row))
        log_likelihoods.append(np.log(multivariate_normal.pdf(row, mean=mu, cov=K)))
        mutant_inds.append(i)

    print(f"Number of mutants which are likely to be WT: {sum(null_hypothesis_results)}")
    print(
        f"Number of mutants which are significantly different to WT: {len(null_hypothesis_results) - sum(null_hypothesis_results)}")

    return log_likelihoods, mutant_inds, null_hypothesis_results


def plot_mu_and_K(K, mu):
    """Show the mean and covariance matrix found using WT data
    """
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].set_title("Covariance matrix")
    im = axs[0].imshow(np.abs(K), norm=LogNorm())
    fig.colorbar(im, ax=axs[0])
    axs[1].set_title("Mean")
    axs[1].plot(mu)
    fig.tight_layout()
    plt.show()


def plot_trained_gp(X, gp, y):
    """Visualise the trained GP mean, variance, and training points
    """

    X_test = np.linspace(-0.1, 1.1, 1000).reshape(-1, 1)
    y_pred, y_std = gp.predict(X_test, return_std=True)

    # Plot the training data points
    plt.fill_between(X_test.ravel(), y_pred - 3 * y_std, y_pred + 3 * y_std, alpha=0.5,
                     label="$\pm 3\sigma$ confidence interval")
    plt.fill_between(X_test.ravel(), y_pred - 2 * y_std, y_pred + 2 * y_std, alpha=0.5,
                     label="$\pm 2\sigma$ confidence interval")
    plt.fill_between(X_test.ravel(), y_pred - y_std, y_pred + y_std, alpha=0.5,
                     label="$\pm \sigma$ confidence interval")
    plt.plot(X_test, y_pred, label="GP mean")
    plt.scatter(X, y, label="Training data", marker="x", color="black", alpha=0.5)
    plt.legend()
    plt.show()


def fit_gp_to_wt(X, wt_series, y):
    """Fit Gaussian Process to WT data.

    If we first extract all the time series corresponding to WT, we want to learn from the uncertainties from these
    ~400 time series which should nominally be identical. We could also add a WhiteNoise kernel to account for the
    variance between these time series.
    """
    wt_std = wt_series.std(axis=0)
    avg_variance = wt_std.mean() ** 2

    # RBF represents a smooth function prior, WhiteKernel represents noise/uncertainty in measurements
    kernel = RBF(length_scale=1) + WhiteKernel(noise_level=avg_variance)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=False)
    gp.fit(X, y)

    # Print the kernel parameters found by the log marginal likelihood maximisation
    print(gp.kernel_)

    return gp


def plot_wt_variance(wt_series):
    """Show the variance of the WT as a function of time point
    """
    wt_std = wt_series.std(axis=0)
    plt.plot(wt_std)
    plt.title("Standard deviation of WT")
    plt.xlabel("Time point")
    plt.show()


def separate_mutants_and_wt(df):
    """Data preparation function to split the data into WT and mutants
    """
    y2_series_df = create_y2_df(df)
    genes_to_indices = construct_gene_to_df_inds_map(df, y2_series_df)
    wt_indices = genes_to_indices[""]
    wt_series = y2_series_df.loc[wt_indices]
    mutant_series = y2_series_df.drop(wt_indices)
    return mutant_series, wt_indices, wt_series


def mahalanobis_distance(x, mean, covariance):
    """The mahalanobis distance is a distance measure between a distribution and a sample point. In this case
    the distribution is hardcoded to Gaussian.
    """
    deviation = x - mean
    inv_covariance = np.linalg.inv(covariance)
    distance = np.sqrt(np.dot(np.dot(deviation.T, inv_covariance), deviation))
    return distance


def null_hypothesis_test(mu, K, sample, significance_level=0.05):
    """Perform a hypothesis test to determine whether the sample was likely to have been generated from the given
    multivariate gaussian distribution"""
    mahalanobis_dist = mahalanobis_distance(sample, mu, K)

    degrees_of_freedom = len(mu)
    critical_value = chi2.ppf(1 - significance_level, df=degrees_of_freedom)

    # Compare Mahalanobis Distance to Critical Value
    if mahalanobis_dist**2 > critical_value:
        return False
    else:
        return True


if __name__ == "__main__":
    main()
