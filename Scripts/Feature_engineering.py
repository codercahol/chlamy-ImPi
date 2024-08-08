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
import localreg 

def extract_features(data):
    features = []  # Initialize list to store extracted features
    
    for i in tqdm(range(len(data))):
        row = data.iloc[i]
        time_series_y2 = row.filter(regex=r'^y2_\d+$').dropna().values.astype(float)
        time_series_ynpq = row.filter(regex=r'^ynpq_\d+$').dropna().values.astype(float)

        # Compute statistical features
        std_deviation_y2 = np.std(time_series_y2)
        skewness_value_y2 = skew(time_series_y2)
        kurtosis_value_y2 = kurtosis(time_series_y2)
        
        std_deviation_ynpq = np.std(time_series_ynpq)
        skewness_value_ynpq = skew(time_series_ynpq)
        kurtosis_value_ynpq = kurtosis(time_series_ynpq)

        # Fit linear regression model to detect trend (slope)
        X_y2 = np.arange(len(time_series_y2)).reshape(-1, 1)
        model = LinearRegression().fit(X_y2, time_series_y2)
        coefficients_y2 = model.coef_
        linear_coef_y2 = coefficients_y2[0]
        
        X_ynpq = np.arange(len(time_series_ynpq)).reshape(-1, 1)
        X_poly_ynpq = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_ynpq)
        model = LinearRegression().fit(X_poly_ynpq, time_series_ynpq)
        coefficients_ynpq = model.coef_
        linear_coef_ynpq = coefficients_ynpq[0]
        quadratic_coef_ynpq = coefficients_ynpq[1] if len(coefficients_ynpq) > 1 else 0.0

        smooth_y2 = local_smooth(time_series_y2, d=1, sigma=12)
        derivative_y2 = np.gradient(smooth_y2)

        smooth_ynpq = local_smooth(time_series_ynpq, d=1, sigma=12)
        derivative_ynpq = np.gradient(smooth_ynpq)

        abs_var_y2 = np.sum(np.abs(derivative_y2))
        abs_var_ynpq = np.sum(np.abs(derivative_ynpq))

        max_slope_y2 = np.max(np.abs(derivative_y2))
        max_slope_ynpq = np.max(np.abs(derivative_ynpq))

        div_y2 = smooth_y2[-1] - smooth_y2[0]
        div_ynpq = smooth_ynpq[-1] - smooth_ynpq[0]

        # Create a dictionary of features for the current row
        row_features = {
            'StdDev_y2': std_deviation_y2,
            'Skewness_y2': skewness_value_y2,
            'Kurtosis_y2': kurtosis_value_y2,
            'Linear_coef_y2': linear_coef_y2,
            'Quadratic_coef_y2': quadratic_coef_y2,
            'abs_var_y2': abs_var_y2,
            'div_y2': div_y2,
            'Max_slope_y2': max_slope_y2,

            'StdDev_ynpq': std_deviation_ynpq,
            'Skewness_ynpq': skewness_value_ynpq,
            'Kurtosis_ynpq': kurtosis_value_ynpq,
            'Linear_coef_ynpq': linear_coef_ynpq,
            'Quadratic_coef_ynpq': quadratic_coef_ynpq,
            'abs_var_ynpq': abs_var_ynpq,
            'div_ynpq': div_ynpq,
            'Max_slope_ynpq': max_slope_ynpq
        }
        
        # Append features to the list
        features.append(row_features)
    
    # Convert the list of dictionaries to a DataFrame
    features_df = pd.DataFrame(features)

    data_with_features = pd.concat([data.reset_index(drop=True), features_df], axis=1)
    
    return data_with_features

# Function to fit piecewise linear regression to identify segments and slopes
def fit_piecewise_linear_regression(time_series, change_points):
    segments = []
    for i in range(len(change_points) - 1):
        start_idx = change_points[i]
        end_idx = change_points[i + 1]
        segment_data = time_series[start_idx:end_idx]
        n_samples = len(segment_data)
        X = np.arange(n_samples).reshape(-1, 1)
        model = LinearRegression().fit(X, segment_data)
        slope = model.coef_[0]
        intercept = model.intercept_
        segments.append({
            'StartIdx': start_idx,
            'EndIdx': end_idx,
            'Slope': slope,
            'Intercept': intercept
        })
    return segments

def plot_segments(time_series_noisy, segment_length=10):
    change_points = [i for i in range(0, len(time_series_noisy), segment_length)] + [len(time_series_noisy)]

    segments = fit_piecewise_linear_regression(time_series_noisy, change_points)

    plt.plot(time_series_noisy)
    for segment in segments:
        start_idx = segment['StartIdx']
        end_idx = segment['EndIdx']
        slope = segment['Slope']
        intercept = segment['Intercept']
        plt.plot(range(start_idx, end_idx+1), slope * np.arange(end_idx+1 - start_idx) + intercept, color='red')

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Piecewise Linear Regression')
    plt.show()

def detect_change_points(time_series, model,  pen):
    data = np.array(time_series)
    model = model  # Use least-squares (l2) model for fitting linear segments
    algo = rpt.Pelt(model=model).fit(data)
    result = algo.predict(pen=pen)  # Adjust 'pen' parameter based on sensitivity
    return result

def kernel_smooth(data, sigma=12.0):
    smoothed_data = gaussian_filter1d(data, sigma)
    return smoothed_data
    
def local_smooth(data, d, sigma):
    x = np.arange(len(data))
    return localreg(x, data, degree=d, kernel=rbf.gaussian, radius=sigma)

def get_slope(data):
    features = []
    for i in range(len(data)):
        row = data.iloc[i]
        time_series_y2 = row.filter(regex=r'^y2_\d+$').dropna().values.astype(float)
        time_series_ynpq = row.filter(regex=r'^ynpq_\d+$').dropna().values.astype(float)
        segment_length = len(time_series_y2)//2
        change_points = [i for i in range(0, len(time_series_y2), segment_length)] + [len(time_series_y2)]
        segments_y2 = fit_piecewise_linear_regression(time_series_y2, change_points)
        segments_ynpq = fit_piecewise_linear_regression(time_series_ynpq, change_points)
        slope_1_y2 = segments_y2[0]['Slope']
        slope_2_y2 = segments_y2[1]['Slope']
        # get slope of the linear regression on the whole ts
        X_y2 = np.arange(len(time_series_y2)).reshape(-1, 1)
        model_y2 = LinearRegression().fit(X_y2, time_series_y2)
        slope_y2 = model_y2.coef_[0]
        slope_1_ynpq = segments_ynpq[0]['Slope']
        slope_2_ynpq = segments_ynpq[1]['Slope']
        X_ynpq = np.arange(len(time_series_ynpq)).reshape(-1, 1)
        model_ynpq = LinearRegression().fit(X_ynpq, time_series_ynpq)
        slope_ynpq = model_ynpq.coef_[0]
        row_features = {
            'slope_1_y2': slope_1_y2,
            'slope_2_y2': slope_2_y2,
            'slope_y2': slope_y2,
            'slope_1_ynpq': slope_1_ynpq,
            'slope_2_ynpq': slope_2_ynpq,
            'slope_ynpq': slope_ynpq
        }
        features.append(row_features)
    features_df = pd.DataFrame(features)
    data_with_features = pd.concat([data.reset_index(drop=True), features_df], axis=1)
    return data_with_features

def get_std(data):
    features = []
    for i in range(len(data)):
        row = data.iloc[i]
        time_series_y2 = row.filter(regex=r'^y2_\d+$').dropna().values.astype(float)
        time_series_ynpq = row.filter(regex=r'^ynpq_\d+$').dropna().values.astype(float)
        std_y2 = np.std(time_series_y2)
        std_ynpq = np.std(time_series_ynpq)
        row_features = {
            'std_y2': std_y2,
            'std_ynpq': std_ynpq
        }
        features.append(row_features)
    features_df = pd.DataFrame(features)
    data_with_features = pd.concat([data.reset_index(drop=True), features_df], axis=1)
    return data_with_features