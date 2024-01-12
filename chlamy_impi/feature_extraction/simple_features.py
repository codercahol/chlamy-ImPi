# In this file we define functions used to extract simple features from time series, such as mean and standard deviation


def mean_y2_value(df):
    """We expect the dataframe to have columns for each y2 value, with one row per well

    :returns: A Series with one row per well, and one column for the mean y2 value
    """
    y2_cols = [col for col in df.columns if col.startswith("y2_")]
    return df[y2_cols].mean(axis=1)


def std_y2_value(df):
    """
    :returns: A Series with one row per well, and one column for the std deviation of y2 value
    """
    y2_cols = [col for col in df.columns if col.startswith("y2_")]
    return df[y2_cols].std(axis=1)


def min_y2_value(df):
    """
    :returns: A Series with one row per well, and one column for the min y2 value
    """
    y2_cols = [col for col in df.columns if col.startswith("y2_")]
    return df[y2_cols].min(axis=1)


def max_y2_value(df):
    """
    :returns: A Series with one row per well, and one column for the max y2 value
    """
    y2_cols = [col for col in df.columns if col.startswith("y2_")]
    return df[y2_cols].max(axis=1)