import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def y2_linear_trend(df):
    """Compute the linear trend of the Y2 time series for each well

    :returns: A Series containing the linear trend for each well
    """
    y2_cols = [col for col in df.columns if col.startswith("y2_")]
    df = df.copy()

    for i, row in df[y2_cols].iterrows():
        row = row.dropna()
        if len(row) == 0:
            df.loc[i, "y2_linear_trend"] = np.nan
            continue

        row = row[:-1]
        df.loc[i, "y2_linear_trend"] = np.polyfit(range(len(row)), row, 1)[0]

    return df["y2_linear_trend"]


def y2_quadratic_trend(df):
    """Compute the quadratic trend of the Y2 time series for each well

    :returns: A Series containing the quadratic trend for each well
    """
    y2_cols = [col for col in df.columns if col.startswith("y2_")]
    df = df.copy()

    for i, row in df[y2_cols].iterrows():
        row = row.dropna()
        if len(row) == 0:
            df.loc[i, "y2_linear_trend"] = np.nan
            continue

        row = row[:-1]
        df.loc[i, "y2_quadratic_trend"] = np.polyfit(range(len(row)), row, 2)[0]

    return df["y2_quadratic_trend"]


def exponential_decay(x, A, tau, C):
    return A * np.exp(-x / tau) + C


def fit_exponential_decay(ts: pd.Series):
    x = np.array(range(len(ts)))
    y = ts.values

    # Initial guess for parameters - start with very flat exponential
    initial_guess = (y[0], 100, 0)

    params, covariance = curve_fit(f=exponential_decay, xdata=x, ydata=y, p0=initial_guess)

    return params, covariance


def y2_exponential_decay_time(df):
    """Compute the time constant of the exponential decay of the Y2 time series for each well

    :returns: A Series containing the time constant for each well
    """
    y2_cols = [col for col in df.columns if col.startswith("y2_")]

    for i, row in df[y2_cols].iterrows():
        row = row.dropna()
        if len(row) == 0:
            df.loc[i, "y2_exponential_decay_time"] = np.nan
            continue
        try:
            params, covariance = fit_exponential_decay(row.T[:-1])  # Skip final entry since it is an outlier
            df.loc[i, "y2_exponential_decay_time"] = params[1]
        except RuntimeError:
            df.loc[i, "y2_exponential_decay_time"] = np.nan
            continue

    return df["y2_exponential_decay_time"]