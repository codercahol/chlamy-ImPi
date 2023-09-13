import pandas as pd


def average_across_strain_replicates(df, param_name):
    """
    Average across strain replicates
    Input:
        df: pandas dataframe
    Output:
        df: pandas dataframe
    """
    df_means = (
        df.groupby(["time", "strain"])
        .mean()
        .reset_index()
        .drop(columns="strain_replicate")
    )
    df_std = (
        df.groupby(["time", "strain"])
        .std()
        .reset_index()
        .drop(columns=["strain_replicate", "WT"])
    )
    df_means.rename(columns={param_name: param_name + "_mean"}, inplace=True)
    df_std.rename(columns={param_name: param_name + "_std"}, inplace=True)
    df = pd.merge(df_means, df_std, on=["time", "strain"])
    df["WT"] = df["WT"].astype(bool)
    return df
