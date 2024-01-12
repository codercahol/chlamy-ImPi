import pandas as pd


def rolling_avg_y2_value(df):
    """Compute a rolling average of the mean Y2 value for each well
    """
    y2_cols = [col for col in df.columns if col.startswith("y2_")]
    return df[y2_cols].T.rolling(window=5, step=5).mean()