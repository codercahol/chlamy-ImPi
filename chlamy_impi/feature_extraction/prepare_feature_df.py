# This file contains functions to extract features from the database, using the functions defined in feature_extraction
# If run as main, will make some plots of features

import pandas as pd
import matplotlib.pyplot as plt

from chlamy_impi.feature_extraction.trend_features import y2_exponential_decay_time, y2_linear_trend, y2_quadratic_trend
from chlamy_impi.feature_extraction.simple_features import mean_y2_value, std_y2_value, min_y2_value, max_y2_value
from chlamy_impi.feature_extraction.rolling_features import rolling_avg_y2_value
from chlamy_impi.normalisation.y2_normalisation import get_normalised_y2_df


def get_y2_features_continuous_light(df):
    """Extract features from the database, assuming that we have continuous light treatment.
    This is a combination of simple features, with features that compare the 20h_HL and 20h_ML treatments for each well

    Note: this approach won't scale very nicely to looking at more than two light treatments
    """
    assert "fv_fm" in df.columns, "fv_fm column not found in dataframe"
    assert "y2_1" in df.columns, "y2_1 column not found in dataframe"

    # First restructure the dataframe so that wells with the same light treatment are in the same row
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        #print(df[(df["plate"] == 2) & (df["mutant_ID"] == "WT")].head())
        pass

    df_hl = df[df["light_regime"] == "20h_HL"]
    df_ml = df[df["light_regime"] == "20h_ML"]

    feature_cols = ["fv_fm"] + [f"y2_{i}" for i in range(1, 42)]
    feature_df_hl = get_simple_y2_value_features(df_hl[feature_cols])
    feature_df_ml = get_simple_y2_value_features(df_ml[feature_cols])

    # Drop the fv/fm column to prevent repetition
    feature_df_hl = feature_df_hl.drop(columns=["fv_fm"])
    feature_df_ml = feature_df_ml.drop(columns=["fv_fm"])

    # Add new columns using the features
    df_hl = pd.concat([df_hl, feature_df_hl], axis=1)
    df_ml = pd.concat([df_ml, feature_df_ml], axis=1)

    # Merge on index
    df_merged = pd.merge(df_hl, df_ml, on=["plate", "i", "j"], suffixes=("_hl", "_ml"), validate="1:1")

    # Finally, put all features into one dataframe
    features = pd.DataFrame()
    features["plate"] = df_merged["plate"]
    features["i"] = df_merged["i"]
    features["j"] = df_merged["j"]
    for col in feature_df_hl.columns:
        features[f"{col}_hl"] = df_merged[f"{col}_hl"]
        features[f"{col}_ml"] = df_merged[f"{col}_ml"]

    print("Constructed features dataframe")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(features.head())

    return features


def get_simple_y2_value_features(df):
    """Extract features from the database - assuming that we just want one feature per row.
    This ignores features which could use more than one row, such as the difference between different light treatments

    :param df: A dataframe with one row per well, and one column for each time point
    :returns: A dataframe with one row per well, and one column for each feature
    """
    assert "fv_fm" in df.columns, "fv_fm column not found in dataframe"
    assert "y2_1" in df.columns, "y2_1 column not found in dataframe"

    features = pd.DataFrame()
    features["fv_fm"] = df["fv_fm"]
    features["mean_y2_value"] = mean_y2_value(df)
    features["std_y2_value"] = std_y2_value(df)
    features["min_y2_value"] = min_y2_value(df)
    features["max_y2_value"] = max_y2_value(df)
    #features["rolling_avg_y2_value"] = rolling_avg_y2_value(df)  # TODO: needs more work to ingest since its a dataframe
    #features["y2_exponential_decay_time"] = y2_exponential_decay_time(df)  # TODO: currently doesnt seem reliable
    features["y2_linear_trend"] = y2_linear_trend(df)
    features["y2_quadratic_trend"] = y2_quadratic_trend(df)

    return features


def normalise_features(features: pd.DataFrame) -> pd.DataFrame:
    """Normalise the features to have zero mean and unit variance

    :param features: A dataframe with one row per well, and one column for each feature
    :returns: A dataframe with one row per well, and one column for each feature
    """
    if ["plate", "i", "j"] == list(features.columns[:3]):
        # Leave plate, i, j columns alone
        features = features.copy()
        features[features.columns[3:]] = (features[features.columns[3:]] - features[features.columns[3:]].mean()) / features[features.columns[3:]].std()
        return features
    else:
        return (features - features.mean()) / features.std()


def main():
    """Make some plots of features
    """

    df = get_normalised_y2_df()

    features = get_simple_y2_value_features(df)

    # Plot histograms of each feature
    num_features = len(features.columns)
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "Computer Modern"

    for i, col in enumerate(features.columns):
        fig, ax = plt.subplots(1, 1, figsize=(6, 8))
        ax.hist(features[col], bins=100)
        ax.set_xlabel(col)
        ax.set_title(col)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.savefig(f"feature_{col}.png")


if __name__ == "__main__":
    main()