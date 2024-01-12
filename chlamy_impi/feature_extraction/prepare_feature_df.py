# This file contains functions to extract features from the database, using the functions defined in feature_extraction
# If run as main, will make some plots of features

import pandas as pd
import matplotlib.pyplot as plt

from chlamy_impi.feature_extraction.trend_features import y2_exponential_decay_time
from chlamy_impi.feature_extraction.simple_features import mean_y2_value, std_y2_value, min_y2_value, max_y2_value
from chlamy_impi.feature_extraction.rolling_features import rolling_avg_y2_value
from chlamy_impi.normalisation.y2_normalisation import get_normalised_y2_df


def get_y2_value_features(df):
    """Extract features from the database

    :param df: A dataframe with one row per well, and one column for each time point
    :returns: A dataframe with one row per well, and one column for each feature
    """

    features = pd.DataFrame()
    features["fv_fm"] = df["fv_fm"]
    features["mean_y2_value"] = mean_y2_value(df)
    features["std_y2_value"] = std_y2_value(df)
    features["min_y2_value"] = min_y2_value(df)
    features["max_y2_value"] = max_y2_value(df)
    #features["rolling_avg_y2_value"] = rolling_avg_y2_value(df)  # TODO: needs more work to ingest since its a dataframe
    features["y2_exponential_decay_time"] = y2_exponential_decay_time(df)

    return features


def main():
    """Make some plots of features
    """

    df = get_normalised_y2_df()

    features = get_y2_value_features(df)

    # Plot histograms of each feature
    num_features = len(features.columns)
    fig, axes = plt.subplots(num_features, 1, figsize=(5, 5 * num_features))
    for i, col in enumerate(features.columns):
        axes[i].hist(features[col])
        axes[i].set_xlabel(col)