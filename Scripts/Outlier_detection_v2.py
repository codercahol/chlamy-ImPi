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
import scipy.stats as stats
from scipy.stats import norm

def folded_normal_cdf(x, mu, sigma_squared):
    # CDF of folded normal distribution with mean mu and variance sigma_squared
    z1 = (x + mu) / np.sqrt(2 * sigma_squared)
    z2 = (x - mu) / np.sqrt(2 * sigma_squared)
    return 0.5 * (erf(z1) + erf(z2))

def median_folded_normal_cdf(x, r, mu, sigma_squared, n):
    # CDF of the median of r i.i.d. folded normal random variables
    F_y = folded_normal_cdf(x, mu, sigma_squared)
    cdf = 0
    for j in range(r, n + 1):
        cdf += comb(n, j) * (F_y**j) * ((1 - F_y)**(n - j))
    return cdf

def compute_norm_reduced(data, cov_matrix, wt_mean):
    data_y2 = data.filter(like='y2_').dropna(axis=1).values
    distances = []
    sqrt_inv = scipy.linalg.sqrtm(np.linalg.inv((1 + np.sqrt(1/3))*cov_matrix))
    for mutant_data in data_y2:
        distance = np.linalg.norm(np.dot(sqrt_inv, mutant_data - wt_mean))**2
        distances.append(distance)
    return distances

def test_mean_zero_gaussian_vector(sample_vector, alpha=0.05):
    # Number of dimensions (length of the sample vector)
    n = len(sample_vector)
    
    # Compute the sample mean
    sample_mean = np.mean(sample_vector)
    quantile = scipy.stats.norm.ppf(1 - alpha)
    # Determine decision
    if np.abs(sample_mean) >= quantile * np.sqrt(1/n):
        return True, np.abs(sample_mean)
    else:
        return False, np.abs(sample_mean)

def test_mean_zero_gaussian_vector_2(sample_vector, alpha):
    # Number of dimensions (length of the sample vector)
    n = len(sample_vector)
    
    # Compute the sample mean
    sample_mean_abs = np.mean(np.abs(sample_vector))

    # get the 1 - alpha ppf of a normal distrib with mean 0 and variance 1
    threshold = scipy.stats.norm.ppf(1 - alpha/2)
    
    # Determine decision
    if np.abs(sample_mean_abs - np.sqrt(2/np.pi)) >= threshold * (1 - 2/np.pi)*np.sqrt(1/n):
        return True
    else:
        return False

def test_chi2(sample_vector, alpha):
    # Number of dimensions (length of the sample vector)
    n = len(sample_vector)

    # Determine decision
    if np.linalg.norm(sample_vector)**2 > chi2.ppf(1 - alpha, n):
        return True, np.linalg.norm(sample_vector)**2
    else:
        return False, np.linalg.norm(sample_vector)**2

def test_divergent_end_down(sample_vector, alpha, w, p):
    # Number of dimensions (length of the sample vector)
    n = len(sample_vector)
    quantile = scipy.stats.norm.ppf(1 - alpha)
    div = np.zeros(n)
    for i in range(n):
        if sample_vector[i] < -quantile :
            div[i] = True
    # check if at the beginning the point are on average not div and at the end they are
    window = w
    if np.sum(div[:window]) < p*window and np.sum(div[-window:]) > p*window :
        return True, -sample_vector[-window:].sum()
    else :
        return False, -sample_vector[-window:].sum()

def test_divergent_end_up(sample_vector, alpha, w, p):
    # Number of dimensions (length of the sample vector)
    n = len(sample_vector)
    quantile = scipy.stats.norm.ppf(1 - alpha)
    div = np.zeros(n)
    for i in range(n):
        if sample_vector[i] > quantile :
            div[i] = True
    # check if at the beginning the point are on average not div and at the end they are
    window = w
    if np.sum(div[:window]) < p*window and np.sum(div[-window:]) > p*window :
        return True, sample_vector[-window:].sum()
    else :
        return False, sample_vector[-window:].sum()

def test_divergent_beg_down(sample_vector, alpha, w, p):
    # Number of dimensions (length of the sample vector)
    n = len(sample_vector)
    quantile = scipy.stats.norm.ppf(1 - alpha)
    div = np.zeros(n)
    for i in range(n):
        if sample_vector[i] < -quantile :
            div[i] = True
    # check if at the beginning the point are on average not div and at the end they are
    window = w
    if np.sum(div[:window]) > p*window and np.sum(div[-window:]) < p*window :
        return True, -sample_vector[:window].sum()
    else :
        return False, -sample_vector[:window].sum()

def test_divergent_beg_up(sample_vector, alpha, w, p):
    # Number of dimensions (length of the sample vector)
    n = len(sample_vector)
    quantile = scipy.stats.norm.ppf(1 - alpha)
    div = np.zeros(n)
    for i in range(n):
        if sample_vector[i] > quantile :
            div[i] = True
    # check if at the beginning the point are on average not div and at the end they are
    window = w
    if np.sum(div[:window]) > p*window and np.sum(div[-window:]) < p*window :
        return True, sample_vector[:window].sum()
    else :
        return False, sample_vector[:window].sum()

def get_alpha_norm_threshold_WT(data_WT_y2s, alpha):
    norms_WT = []
    for data_WT_y2 in data_WT_y2s:
        norms_WT.append(np.linalg.norm(data_WT_y2))
    # get the 1 - alpha threshold percentile of the norms_WT
    return np.percentile(norms_WT, 100*(1 - alpha))

def compute_point_median(data):
    data_y2 = data.filter(like='y2_').dropna(axis=1).values
    n = len(data_y2[0])
    wt_mean = np.zeros(n)
    medians = []
    for mutant_data in data_y2:
        distances = []
        for i in range(len(wt_mean)):
            distances.append(abs(mutant_data[i] - wt_mean[i]))
        medians.append(np.median(distances))
    return medians

def mean_cov_y2_WT(data, light, plot):
    data_WT_y2 = data[(data['mutant_ID'] == 'WT') & (data['light_regime'] == light)].filter(like='y2_').dropna(axis=1).values
    # Calculate mean and covariance matrix of the WT time series
    wt_mean = np.mean(data_WT_y2, axis=0)
    wt_cov = np.cov(data_WT_y2, rowvar=False)

    # Generate time points
    time_points = np.arange(0, len(wt_mean))

    # Calculate standard deviation (square root of diagonal elements of covariance matrix)
    wt_std = np.sqrt(np.diag(wt_cov))

    if plot:
        # Plot confidence interval (e.g., 99% confidence interval)
        plt.fill_between(time_points, wt_mean - 2.58 * wt_std, wt_mean + 2.58 * wt_std, color='blue', alpha=0.6, label='99% Confidence Interval')

        # plt.fill_between(time_points, wt_mean - 1.96 * wt_std, wt_mean + 1.96 * wt_std, color='blue', alpha=0.6, label='95% Confidence Interval')

        for y2 in data_WT_y2:
            plt.plot(time_points, y2, color = 'gray', alpha = 0.2)

        # Plot mean trajectory
        plt.plot(time_points, wt_mean, label='Mean Trajectory', color='blue')

        # Customize plot
        plt.xlabel('Time')
        plt.ylabel('Y(II) Response')
        plt.title('Mean Trajectory of Wild Type in ' + light + ' with Confidence Interval')
        plt.legend()
        plt.grid(True)

        # Show plot
        plt.show()

    return wt_mean, wt_cov

def mean_cov_y2_WT_v2(data, light, plot):
    data_WT_99_y2 = data[(data['mutant_ID'] == 'WT') & (data['plate'] == 99) & (data['light_regime'] == light)].filter(like='y2_').dropna(axis=1)
    data_WT_not_99_y2 = data[(data['mutant_ID'] == 'WT') & (data['plate'] != 99) & (data['light_regime'] == light)].filter(like='y2_').dropna(axis=1)
    data_WT_y2 = np.concatenate(((1/np.sqrt(1 - 1/383))*data_WT_99_y2, np.sqrt(3/2)*data_WT_not_99_y2), axis=0)
    # Calculate mean and covariance matrix of the WT time series
    wt_mean = np.mean(data_WT_y2, axis=0)
    wt_cov = np.cov(data_WT_y2, rowvar=False)

    # Generate time points
    time_points = np.arange(0, len(wt_mean))

    # Calculate standard deviation (square root of diagonal elements of covariance matrix)
    wt_std = np.sqrt(np.diag(wt_cov))

    if plot:
        # Plot confidence interval (e.g., 99% confidence interval)
        plt.fill_between(time_points, wt_mean - 2.58 * wt_std, wt_mean + 2.58 * wt_std, color='blue', alpha=0.6, label='99% Confidence Interval')

        # plt.fill_between(time_points, wt_mean - 1.96 * wt_std, wt_mean + 1.96 * wt_std, color='blue', alpha=0.6, label='95% Confidence Interval')

        for y2 in data_WT_y2:
            plt.plot(time_points, y2, color = 'gray', alpha = 0.2)

        # Plot mean trajectory
        plt.plot(time_points, wt_mean, label='Mean Trajectory', color='blue')

        # Customize plot
        plt.xlabel('Time')
        plt.ylabel('Y(II) Response')
        plt.title('Mean Trajectory of Wild Type in ' + light + ' with Confidence Interval')
        plt.legend()
        plt.grid(True)

        # Show plot
        plt.show()

    return wt_mean, wt_cov

def test_cov(data, light):
    data_WT_99_y2 = data[(data['mutant_ID'] == 'WT') & (data['plate'] == 99) & (data['light_regime'] == light)].filter(like='y2_').dropna(axis=1)
    wt_cov = np.cov(data_WT_99_y2, rowvar=False)
    return wt_cov

def mean_cov_ynpq_WT_v2(data, light, plot):
    data_WT_99_ynpq = data[(data['mutant_ID'] == 'WT') & (data['plate'] == 99) & (data['light_regime'] == light)].filter(like='ynpq_').dropna(axis=1).values
    data_WT_not_99_ynpq = data[(data['mutant_ID'] == 'WT') & (data['plate'] != 99) & (data['light_regime'] == light)].filter(like='ynpq_').dropna(axis=1).values
    data_WT_ynpq = np.concatenate(((1/np.sqrt(1 - 1/383))*data_WT_99_ynpq, np.sqrt(3/2)*data_WT_not_99_ynpq), axis=0)
    # Calculate mean and covariance matrix of the WT time series
    wt_mean_ynpq = np.mean(data_WT_ynpq, axis=0)
    wt_cov_ynpq = np.cov(data_WT_ynpq, rowvar=False)

    return wt_mean_ynpq, wt_cov_ynpq

def set_sigma_norm(data, cov_matrices):
    for light in data['light_regime'].unique():
        data_light = data[data['light_regime'] == light]
        n = ((1/2)*data_light['num_frames'].values[0] - 2).astype(int)
        distances = compute_sigma_distances(data_light, cov_matrices[light], np.zeros(n))
        data.loc[data['light_regime'] == light, 'euclidian_distances'] = distances

def compute_euclidian_distances(data, wt_mean):
    data_y2 = data.filter(like='y2_').dropna(axis=1).values
    distances = []
    for mutant_data in data_y2:
        distance = np.linalg.norm(mutant_data - wt_mean)**2
        distances.append(distance)
    return distances

def set_euclidian_norm(data):
    for light in data['light_regime'].unique():
        data_light = data[data['light_regime'] == light]
        n = ((1/2)*data_light['num_frames'].values[0] - 2).astype(int)
        distances = compute_euclidian_distances(data_light, np.zeros(n))
        data.loc[data['light_regime'] == light, 'euclidian_distances'] = distances

def set_median_dev(data):
    for light in data['light_regime'].unique():
        data_light = data[data['light_regime'] == light]
        n = ((1/2)*data_light['num_frames'].values[0] - 2).astype(int)
        medians = compute_point_median(data_light, np.zeros(n))
        data.loc[data['light_regime'] == light, 'median'] = medians

def get_cov_norm(data):
    sigma_dict = {}
    for light in data['light_regime'].unique():
        data_light = data[data['light_regime'] == light]
        wt_mean, wt_cov = mean_cov_y2_WT_v2(data_light, light, plot=False)
        sigma_dict[light] = wt_cov
    return sigma_dict

def set_sigma_gene(data, sigma_dict):
    for light in sigma_dict.keys():
        data.loc[data['light_regime'] == light, 'sigma'] = sigma_dict[light]

def set_log_likelihood_norm(data):
    for i in range(len(data)):
        # Parameters
        row = data.iloc[i]
        sample_vector = row.filter(like='y2_').dropna().values
        n = len(sample_vector)  # Dimensionality of the vector
        v = row['sigma']**2  # Known variance of the vector

        # Calculate log likelihood ratio for likelihood ratio test
        log_likelihood_null = multivariate_normal.logpdf(sample_vector, mean=np.zeros(n), cov=np.diag(np.full(n, v)))
        data['log_likelihood_null'].iloc[i] = log_likelihood_null

def set_pvalues(data):
    for i in range(len(data)):
        data['p_value'].iloc[i] = 1 - chi2.cdf(data['euclidian_distances'].iloc[i], ((1/2)*data['num_frames'].iloc[i] - 2))

def detect_outliers_p_value(data, alpha) :
    # set outlier to true if data['distances']*data['sigma'] is greater than the 95th percentile of the chi-squared distribution
    data['outlier_euclidian_distance'] = data['p_value'] < alpha

def detect_outliers_euclidian_distance(data, alpha) :
    # set outlier to true if data['distances']*data['sigma'] is greater than the 95th percentile of the chi-squared distribution
    set_euclidian_norm(data)
    set_pvalues(data)
    data['outlier_euclidian_distance'] = data['p_value'] < alpha

def detect_outlier_mean_unreduced(data, alpha, cov_matrices=None) :
    for light in data['light_regime'].unique():
        data_light = data[data['light_regime'] == light]
        data_light_y2 = data_light.filter(like='y2_').dropna(axis=1).values
        if cov_matrices == None:
            wt_mean, wt_cov = mean_cov_y2_WT_v2(data_light, light, plot=False)
        else :
            wt_cov = cov_matrices[light]
        # wt_std = np.sqrt(np.mean(np.diag(wt_cov)))
        sqrt_inv = scipy.linalg.sqrtm(np.linalg.inv(np.diag(np.diag((3/2)*wt_cov))))
        # sqrt_inv = 1/wt_std
        data_light_y2_norm_std = np.dot(data_light_y2, sqrt_inv)
        results = []
        means = []
        for i in range(data_light_y2_norm_std.shape[0]):
            result, mean = test_mean_zero_gaussian_vector(data_light_y2_norm_std[i], alpha)
            results.append(result)
            means.append(mean)
        data.loc[data['light_regime'] == light, 'outlier_mean_unreduced'] = results
        data.loc[data['light_regime'] == light, 'mean_unreduced'] = means

def detect_outlier_mean_reduced(data, alpha, cov_matrices=None) :
    for light in data['light_regime'].unique():
        data_light = data[data['light_regime'] == light]
        data_light_y2 = data_light.filter(like='y2_').dropna(axis=1).values
        if cov_matrices == None:
            wt_mean, wt_cov = mean_cov_y2_WT_v2(data_light, light, plot=False)
        else :
            wt_cov = cov_matrices[light]
        sqrt_inv = scipy.linalg.sqrtm(np.linalg.inv((3/2)*wt_cov))
        data_light_y2_norm_std = np.dot(data_light_y2, sqrt_inv)
        results = []
        means = []
        for i in range(data_light_y2_norm_std.shape[0]):
            result, mean = test_mean_zero_gaussian_vector(data_light_y2_norm_std[i], alpha)
            results.append(result)
            means.append(mean)
        data.loc[data['light_regime'] == light, 'outlier_mean_reduced'] = results
        data.loc[data['light_regime'] == light, 'mean_reduced'] = means

def detect_outlier_mean_reduced_v2(data, alpha, cov_matrices_y2=None, cov_matrices_ynpq=None) :
    for light in data['light_regime'].unique():
        data_light = data[data['light_regime'] == light]
        data_light_y2 = data_light.filter(like='y2_').dropna(axis=1).values
        data_light_ynpq = data_light.filter(like='ynpq_').dropna(axis=1).values
        if cov_matrices_y2 == None:
            wt_mean_y2, wt_cov_y2 = mean_cov_y2_WT_v2(data_light, light, plot=False)
            wt_mean_ynpq, wt_cov_ynpq = mean_cov_ynpq_WT_v2(data_light, light, plot=False)
        else :
            wt_cov_y2 = cov_matrices_y2[light]
            wt_cov_ynpq = cov_matrices_ynpq[light]
        sqrt_inv_eps_y2 = scipy.linalg.sqrtm(np.linalg.inv(wt_cov_y2))
        sqrt_inv_eps_ynpq = scipy.linalg.sqrtm(np.linalg.inv(wt_cov_ynpq))
        results_y2 = []
        results_ynpq = []
        means_y2 = []
        means_ynpq = []
        for i in range(data_light_y2.shape[0]):
            if data_light.iloc[i]['mutant_ID'] == 'WT' and data_light.iloc[i]['plate'] == 99:
                sqrt_inv_y2 = (1/np.sqrt(1-1/384))*sqrt_inv_eps_y2
                sqrt_inv_ynpq = (1/np.sqrt(1-1/384))*sqrt_inv_eps_ynpq
            elif data_light.iloc[i]['mutant_ID'] == 'WT' and data_light.iloc[i]['plate'] != 99:
                sqrt_inv_y2 = np.sqrt(3/2)*sqrt_inv_eps_y2
                sqrt_inv_ynpq = np.sqrt(3/2)*sqrt_inv_eps_ynpq
            else :
                sqrt_inv_y2 = np.sqrt(3/4)*sqrt_inv_eps_y2
                sqrt_inv_ynpq = np.sqrt(3/4)*sqrt_inv_eps_ynpq
            data_light_y2_norm_std = np.dot(data_light_y2[i], sqrt_inv_y2)
            data_light_ynpq_norm_std = np.dot(data_light_ynpq[i], sqrt_inv_ynpq)
            result_y2, mean_y2 = test_mean_zero_gaussian_vector(data_light_y2_norm_std, alpha)
            result_ynpq, mean_ynpq = test_mean_zero_gaussian_vector(data_light_ynpq_norm_std, alpha)
            results_y2.append(result_y2)
            results_ynpq.append(result_ynpq)
            means_y2.append(mean_y2)
            means_ynpq.append(mean_ynpq)
        data.loc[data['light_regime'] == light, 'outlier_mean_reduced_v2_y2'] = results_y2
        data.loc[data['light_regime'] == light, 'outlier_mean_reduced_v2_ynpq'] = results_ynpq
        data.loc[data['light_regime'] == light, 'mean_reduced_v2_y2'] = means_y2
        data.loc[data['light_regime'] == light, 'mean_reduced_v2_ynpq'] = means_ynpq

def detect_outlier_chi2_unreduced(data, alpha, cov_matrices=None) :
    for light in data['light_regime'].unique():
        data_light = data[data['light_regime'] == light]
        data_light_y2 = data_light.filter(like='y2_').dropna(axis=1).values
        if cov_matrices == None:
            wt_mean, wt_cov = mean_cov_y2_WT_v2(data_light, light, plot=False)
        else :
            wt_cov = cov_matrices[light]
        sqrt_inv = scipy.linalg.sqrtm(np.linalg.inv(np.diag(np.diag((3/2)*wt_cov))))
        data_light_y2_norm_std = np.dot(data_light_y2, sqrt_inv)
        results = []
        norms = []
        for i in range(data_light_y2_norm_std.shape[0]):
            result, norm = test_chi2(data_light_y2_norm_std[i], alpha)
            results.append(result)
            norms.append(norm)
        data.loc[data['light_regime'] == light, 'outlier_chi2_unreduced'] = results
        data.loc[data['light_regime'] == light, 'norm_unreduced'] = norms

def detect_outlier_chi2_unreduced_v2(data, alpha, cov_matrices_y2=None, cov_matrices_ynpq=None) :
    for light in data['light_regime'].unique():
        data_light = data[data['light_regime'] == light]
        data_light_y2 = data_light.filter(like='y2_').dropna(axis=1).values
        data_light_ynpq = data_light.filter(like='ynpq_').dropna(axis=1).values
        if cov_matrices_y2 == None:
            wt_mean, wt_cov_y2 = mean_cov_y2_WT_v2(data_light, light, plot=False)
            wt_mean, wt_cov_ynpq = mean_cov_ynpq_WT_v2(data_light, light, plot=False)
        else :
            wt_cov_y2 = cov_matrices_y2[light]
        sqrt_inv_eps_y2 = scipy.linalg.sqrtm(np.linalg.inv(np.diag(np.diag(wt_cov_y2))))
        results = []
        norms = []
        for i in range(data_light_y2.shape[0]):
            if data_light.iloc[i]['mutant_ID'] == 'WT' and data_light.iloc[i]['plate'] == 99:
                sqrt_inv = (1/np.sqrt(1-1/383))*sqrt_inv_eps
            elif data_light.iloc[i]['mutant_ID'] == 'WT' and data_light.iloc[i]['plate'] != 99:
                sqrt_inv = np.sqrt(3/2)*sqrt_inv_eps
            else :
                sqrt_inv = np.sqrt(3/4)*sqrt_inv_eps
            data_light_y2_norm_std = np.dot(data_light_y2[i], sqrt_inv)
            result, norm = test_chi2(data_light_y2_norm_std, alpha)
            results.append(result)
            norms.append(norm)
        data.loc[data['light_regime'] == light, 'outlier_chi2_unreduced_v2'] = results
        data.loc[data['light_regime'] == light, 'norm_unreduced_v2'] = norms

def detect_outlier_chi2_reduced(data, alpha, cov_matrices=None) :
    for light in data['light_regime'].unique():
        data_light = data[data['light_regime'] == light]
        data_light_y2 = data_light.filter(like='y2_').dropna(axis=1).values
        if cov_matrices == None:
            wt_mean, wt_cov = mean_cov_y2_WT_v2(data_light, light, plot=False)
        else :
            wt_cov = cov_matrices[light]
        sqrt_inv = scipy.linalg.sqrtm(np.linalg.inv((3/2)*wt_cov))
        data_light_y2_norm_std = np.dot(data_light_y2, sqrt_inv)
        results = []
        norms = []
        for i in range(data_light_y2_norm_std.shape[0]):
            result, norm = test_chi2(data_light_y2_norm_std[i], alpha)
            results.append(result)
            norms.append(norm)
        data.loc[data['light_regime'] == light, 'outlier_chi2_reduced'] = results
        data.loc[data['light_regime'] == light, 'norm_reduced'] = norms

def detect_outlier_chi2_reduced_v2(data, alpha, cov_matrices=None) :
    for light in data['light_regime'].unique():
        data_light = data[data['light_regime'] == light]
        data_light_y2 = data_light.filter(like='y2_').dropna(axis=1).values
        if cov_matrices == None:
            wt_mean, wt_cov = mean_cov_y2_WT_v2(data_light, light, plot=False)
        else :
            wt_cov = cov_matrices[light]
        sqrt_inv_eps = scipy.linalg.sqrtm(np.linalg.inv(wt_cov))
        results = []
        norms = []
        for i in range(data_light_y2.shape[0]):
            if data_light.iloc[i]['mutant_ID'] == 'WT' and data_light.iloc[i]['plate'] == 99:
                sqrt_inv = (1/np.sqrt(1-1/383))*sqrt_inv_eps
            elif data_light.iloc[i]['mutant_ID'] == 'WT' and data_light.iloc[i]['plate'] != 99:
                sqrt_inv = np.sqrt(3/2)*sqrt_inv_eps
            else :
                sqrt_inv = np.sqrt(3/4)*sqrt_inv_eps
            data_light_y2_norm_std = np.dot(data_light_y2[i], sqrt_inv)
            result, norm = test_chi2(data_light_y2_norm_std, alpha)
            results.append(result)
            norms.append(norm)
        data.loc[data['light_regime'] == light, 'outlier_chi2_reduced_v2'] = results
        data.loc[data['light_regime'] == light, 'norm_reduced_v2'] = norms

def detect_outlier_norm_non_parametric(data, alpha) :
    for light in data['light_regime'].unique():
        data_light = data[data['light_regime'] == light]
        data_light_y2s = data_light.filter(like='y2_').dropna(axis=1).values
        data_light_WT_y2s = data_light[data_light['mutant_ID'] == 'WT'].filter(like='y2_').dropna(axis=1).values
        threshold = get_alpha_norm_threshold_WT(data_light_WT_y2s, alpha)
        results = []
        norms = []
        for data_light_y2 in data_light_y2s :
            norm = np.linalg.norm(data_light_y2)
            if norm > threshold :
                results.append(True)
            else :
                results.append(False)
            norms.append(norm)
        data.loc[data['light_regime'] == light, 'outlier_norm_non_parametric'] = results
        data.loc[data['light_regime'] == light, 'norm'] = norms

def detect_outliers_median(data, alpha) :
    # set outlier to true if data['median'] is greater than the 95th percentile of the median distribution
    list_outlier = []
    mu = 0
    for i in range(len(data)):
        x = data['median'].iloc[i]
        n = ((1/2)*data['num_frames'].iloc[i] - 2).astype(int)
        r = n//2
        sigma_squared = data['sigma'].iloc[i]**2
        proba = 1 - median_folded_normal_cdf(x, r, mu, sigma_squared)
        list_outlier.append(proba < alpha)
    data['outlier_median_unreduced'] = list_outlier

def detect_outlier_median_unreduced(data, alpha, cov_matrices = None):
    for light in data['light_regime'].unique():
        data_light = data[data['light_regime'] == light]
        n = ((1/2)*data_light['num_frames'].iloc[0] - 2).astype(int)
        r = n//2
        medians = compute_point_median(data_light)
        mu = 0
        if cov_matrices == None:
            wt_mean, wt_cov = mean_cov_y2_WT_v2(data_light, light, plot=False)
        else :
            wt_cov = cov_matrices[light]
        sigma_squared = np.mean(np.diag(2*wt_cov))
        results = []
        for i in range(len(medians)):
            proba = 1 - median_folded_normal_cdf(medians[i], r, mu, sigma_squared, n)
            results.append(proba < alpha)
        data.loc[data['light_regime'] == light, 'outlier_median_unreduced'] = results
        data.loc[data['light_regime'] == light, 'median_unreduced'] = medians
        

def detect_outliers_log_likelihood(data, alpha) :
    # set outlier to true if data['log_likelihood_null'] is less than the 95th percentile of the chi-squared distribution
    threshold_80 = chi2.ppf(1 - alpha, 80)
    threshold_40 = chi2.ppf(1 - alpha, 40)
    # set data['threshold'] to threshold_80 if data['num_frames'] is greater than 84, else set it to threshold_40
    data['threshold'] = np.where(data['num_frames'] > 84, threshold_80, threshold_40)
    data['outlier'] = data['log_likelihood_null'] < data['threshold']

# def plot_least_and_most_likely(data):

def detect_diverging_outliers(data, alpha, w, p, cov_matrices_y2=None, cov_matrices_ynpq=None):
    for light in data['light_regime'].unique():
        data_light = data[data['light_regime'] == light]
        data_light_y2 = data_light.filter(like='y2_').dropna(axis=1).values
        data_light_ynpq = data_light.filter(like='ynpq_').dropna(axis=1).values
        if cov_matrices_y2 == None:
            wt_mean_y2, wt_cov_y2 = mean_cov_y2_WT_v2(data_light, light, plot=False)
            wt_mean_ynpq, wt_cov_ynpq = mean_cov_ynpq_WT_v2(data_light, light, plot=False)
        else :
            wt_cov_y2 = cov_matrices_y2[light]
            wt_cov_ynpq = cov_matrices_ynpq[light]
        sqrt_inv_eps_y2 = scipy.linalg.sqrtm(np.linalg.inv(np.diag(np.diag(wt_cov_y2))))
        sqrt_inv_eps_ynpq = scipy.linalg.sqrtm(np.linalg.inv(np.diag(np.diag(wt_cov_ynpq))))
        results_beg_up_y2 = []
        results_beg_up_ynpq = []
        results_beg_down_y2 = []
        results_beg_down_ynpq = []
        results_end_up_y2 = []
        results_end_up_ynpq = []
        results_end_down_y2 = []
        results_end_down_ynpq = []
        sums_beg_up_y2 = []
        sums_beg_up_ynpq = []
        sums_beg_down_y2 = []
        sums_beg_down_ynpq = []
        sums_end_up_y2 = []
        sums_end_up_ynpq = []
        sums_end_down_y2 = []
        sums_end_down_ynpq = []
        for i in range(data_light_y2.shape[0]):
            if data_light.iloc[i]['mutant_ID'] == 'WT' and data_light.iloc[i]['plate'] == 99:
                sqrt_inv_y2 = (1/np.sqrt(1-1/384))*sqrt_inv_eps_y2
                sqrt_inv_ynpq = (1/np.sqrt(1-1/384))*sqrt_inv_eps_ynpq
            elif data_light.iloc[i]['mutant_ID'] == 'WT' and data_light.iloc[i]['plate'] != 99:
                sqrt_inv_y2 = np.sqrt(3/2)*sqrt_inv_eps_y2
                sqrt_inv_ynpq = np.sqrt(3/2)*sqrt_inv_eps_ynpq
            else :
                sqrt_inv_y2 = np.sqrt(3/4)*sqrt_inv_eps_y2
                sqrt_inv_ynpq = np.sqrt(3/4)*sqrt_inv_eps_ynpq
            data_light_y2_norm_std = np.dot(data_light_y2[i], sqrt_inv_y2)
            data_light_ynpq_norm_std = np.dot(data_light_ynpq[i], sqrt_inv_ynpq)
            result_beg_up_y2, sum_beg_up_y2 = test_divergent_beg_up(data_light_y2_norm_std, alpha, w, p)
            result_beg_down_y2, sum_beg_down_y2 = test_divergent_beg_down(data_light_y2_norm_std, alpha, w, p)
            result_end_up_y2, sum_end_up_y2 = test_divergent_end_up(data_light_y2_norm_std, alpha, w, p)
            result_end_down_y2, sum_end_down_y2 = test_divergent_end_down(data_light_y2_norm_std, alpha, w, p)
            results_beg_up_y2.append(result_beg_up_y2)
            results_beg_down_y2.append(result_beg_down_y2)
            results_end_up_y2.append(result_end_up_y2)
            results_end_down_y2.append(result_end_down_y2)
            sums_beg_up_y2.append(sum_beg_up_y2)
            sums_beg_down_y2.append(sum_beg_down_y2)
            sums_end_up_y2.append(sum_end_up_y2)
            sums_end_down_y2.append(sum_end_down_y2)
            result_beg_up_ynpq, sum_beg_up_ynpq = test_divergent_beg_up(data_light_ynpq_norm_std, alpha, w, p)
            result_beg_down_ynpq, sum_beg_down_ynpq = test_divergent_beg_down(data_light_ynpq_norm_std, alpha, w, p)
            result_end_up_ynpq, sum_end_up_ynpq = test_divergent_end_up(data_light_ynpq_norm_std, alpha, w, p)
            result_end_down_ynpq, sum_end_down_ynpq = test_divergent_end_down(data_light_ynpq_norm_std, alpha, w, p)
            results_beg_up_ynpq.append(result_beg_up_ynpq)
            results_beg_down_ynpq.append(result_beg_down_ynpq)
            results_end_up_ynpq.append(result_end_up_ynpq)
            results_end_down_ynpq.append(result_end_down_ynpq)
            sums_beg_up_ynpq.append(sum_beg_up_ynpq)
            sums_beg_down_ynpq.append(sum_beg_down_ynpq)
            sums_end_up_ynpq.append(sum_end_up_ynpq)
            sums_end_down_ynpq.append(sum_end_down_ynpq)
        data.loc[data['light_regime'] == light, 'outlier_divergent_beg_up_y2'] = results_beg_up_y2
        data.loc[data['light_regime'] == light, 'sum_beg_up_y2'] = sums_beg_up_y2
        data.loc[data['light_regime'] == light, 'outlier_divergent_beg_down_y2'] = results_beg_down_y2
        data.loc[data['light_regime'] == light, 'sum_beg_down_y2'] = sums_beg_down_y2
        data.loc[data['light_regime'] == light, 'outlier_divergent_end_up_y2'] = results_end_up_y2
        data.loc[data['light_regime'] == light, 'sum_end_up_y2'] = sums_end_up_y2
        data.loc[data['light_regime'] == light, 'outlier_divergent_end_down_y2'] = results_end_down_y2
        data.loc[data['light_regime'] == light, 'sum_end_down_y2'] = sums_end_down_y2
        data.loc[data['light_regime'] == light, 'outlier_divergent_beg_up_ynpq'] = results_beg_up_ynpq
        data.loc[data['light_regime'] == light, 'sum_beg_up_ynpq'] = sums_beg_up_ynpq
        data.loc[data['light_regime'] == light, 'outlier_divergent_beg_down_ynpq'] = results_beg_down_ynpq
        data.loc[data['light_regime'] == light, 'sum_beg_down_ynpq'] = sums_beg_down_ynpq
        data.loc[data['light_regime'] == light, 'outlier_divergent_end_up_ynpq'] = results_end_up_ynpq
        data.loc[data['light_regime'] == light, 'sum_end_up_ynpq'] = sums_end_up_ynpq
        data.loc[data['light_regime'] == light, 'outlier_divergent_end_down_ynpq'] = results_end_down_ynpq
        data.loc[data['light_regime'] == light, 'sum_end_down_ynpq'] = sums_end_down_ynpq

def detect_outliers_fv_fm_ynpq_end(data, alpha):
    fv_fm = data[(data['mutant_ID'] == 'WT') & (data['flag_y2'] == 'ok')]['fv_fm'].values
    std_fv_fm = np.std(fv_fm)
    quantile = scipy.stats.norm.ppf(1 - alpha/2)
    data['outlier_fv_fm'] = np.where(np.abs(data['fv_fm']) > quantile*std_fv_fm, True, False)
    for light in data['light_regime'].unique():
        data_light = data[data['light_regime'] == light]
        fv_fm_end = data_light[(data_light['mutant_ID'] == 'WT') & (data_light['flag_y2'] == 'ok')]['fv_fm_end'].values
        ynpqend = data_light[(data_light['mutant_ID'] == 'WT') & (data_light['flag_ynpq'] == 'ok')]['ynpqend'].values
        std_fv_fm_end = np.std(fv_fm_end)
        std_ynpqend = np.std(ynpqend)
        data_light['outlier_fv_fm_end'] = np.where(np.abs(data_light['fv_fm_end']) > quantile*std_fv_fm_end, True, False)
        data_light['outlier_ynpqend'] = np.where(np.abs(data_light['ynpqend']) > quantile*std_ynpqend, True, False)
        data.loc[data['light_regime'] == light, 'outlier_fv_fm_end'] = data_light['outlier_fv_fm_end']
        data.loc[data['light_regime'] == light, 'outlier_ynpqend'] = data_light['outlier_ynpqend']

def detect_outliers_abs_var_y2(data, alpha):
    for light in data['light_regime'].unique():
        data_light = data[data['light_regime'] == light]
        abs_var_y2_WT = data_light[(data_light['mutant_ID'] == 'WT') & (data_light['flag_y2'] == 'ok')]['abs_var_y2'].values
        shape_ln, loc_ln, scale_ln = stats.lognorm.fit(abs_var_y2_WT)
        # get the quantile of the 95th percentile of the lognormal distribution with the estimated parameters
        quantile = stats.lognorm.ppf(1 - alpha, shape_ln, loc_ln, scale_ln)
        data_light['outlier_abs_var_y2'] = np.where(data_light['abs_var_y2'] > quantile, True, False)
        data.loc[data['light_regime'] == light, 'outlier_abs_var_y2'] = data_light['outlier_abs_var_y2']

def set_mean_group(data):
    data_copy = data.copy()
    # get an interval where 95% of the WT are
    wt_mean_y2 = data_copy[data_copy['mutant_ID'] == 'WT']['mean_y2'].values
    wt_mean_ynpq = data_copy[data_copy['mutant_ID'] == 'WT']['mean_ynpq'].values

    # get the 95% interval
    interval_y2 = np.percentile(wt_mean_y2, [2.5, 97.5])
    interval_ynpq = np.percentile(wt_mean_ynpq, [2.5, 97.5])

    data_copy['mean_cluster_y2'] = 0
    data_copy['mean_cluster_ynpq'] = 0

    for idx in data_copy.index:
        if data_copy.loc[idx, 'mean_y2'] < interval_y2[0]:
            data_copy.loc[idx, 'mean_cluster_y2'] = -1
        elif data_copy.loc[idx, 'mean_y2'] >= interval_y2[1]:
            data_copy.loc[idx, 'mean_cluster_y2'] = 1
        
        if data_copy.loc[idx, 'mean_ynpq'] < interval_ynpq[0]:
            data_copy.loc[idx, 'mean_cluster_ynpq'] = -1
        elif data_copy.loc[idx, 'mean_ynpq'] >= interval_ynpq[1]:
            data_copy.loc[idx, 'mean_cluster_ynpq'] = 1

    return data_copy

def detect_and_cluster_slopes(data, alpha):
    data_cluster = pd.DataFrame()

    for light in data['light_regime'].unique():
        data_light = data[data['light_regime'] == light].reset_index(drop=True)
        slope_y2_WT = data[(data['light_regime'] == light) & (data['mutant_ID'] == 'WT')]['slope_y2'].values
        # get the 95% interval for a normal distribution
        mean_y2, std_y2 = norm.fit(slope_y2_WT)
        interval_y2_norm = norm.interval(1-alpha, loc=mean_y2, scale=std_y2)
        interval_y2 = np.percentile(slope_y2_WT, [(alpha/2)*100, 100 - (alpha/2)*100])
        slope_ynpq_WT = data[(data['light_regime'] == light) & (data['mutant_ID'] == 'WT')]['slope_ynpq'].values
        # get the 95% interval for a normal distribution
        mean_ynpq, std_ynpq = norm.fit(slope_ynpq_WT)
        interval_ynpq_norm = norm.interval(1-alpha, loc=mean_ynpq, scale=std_ynpq)
        interval_ynpq = np.percentile(slope_ynpq_WT, [(alpha/2)*100, 100 - (alpha/2)*100])

        data_light['slope_cluster_y2'] = 0
        data_light['slope_cluster_ynpq'] = 0

        for idx in data_light.index:
            if data_light.loc[idx, 'slope_y2'] < interval_y2_norm[0]:
                data_light.loc[idx, 'slope_cluster_y2'] = -1
            elif data_light.loc[idx, 'slope_y2'] >= interval_y2_norm[1]:
                data_light.loc[idx, 'slope_cluster_y2'] = 1
            
            if data_light.loc[idx, 'slope_ynpq'] < interval_ynpq_norm[0]:
                data_light.loc[idx, 'slope_cluster_ynpq'] = -1
            elif data_light.loc[idx, 'slope_ynpq'] >= interval_ynpq_norm[1]:
                data_light.loc[idx, 'slope_cluster_ynpq'] = 1

        data_light = set_mean_group(data_light)

        data_cluster = pd.concat([data_cluster, data_light])

    return data_cluster

