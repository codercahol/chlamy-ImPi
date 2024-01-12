import pandas as pd
from matplotlib import pyplot as plt

from chlamy_impi.paths import read_df_from_parquet


def get_y2_df() -> pd.DataFrame:
    df = read_df_from_parquet()

    # Drop time columns except first one
    time_cols = [f"measurement_time_{i}" for i in range(1, 82)]
    df = df.drop(columns=time_cols)

    # Drop npq columns
    npq_cols = [f"npq_{i}" for i in range(1, 82)]
    df = df.drop(columns=npq_cols)

    y2_cols = [f"y2_{i}" for i in range(1, 82)] + ["fv_fm"]
    df_no_y2 = df.drop(columns=y2_cols)

    # Check that there are no nan values in the non-measurement columns
    # Some nan values in the y2 may still exist due to blank wells or missing time points
    assert len(df_no_y2[df_no_y2.isna().any(axis=1)]) == 0

    return df


def get_normalised_y2_df() -> pd.DataFrame:
    """Normalise Y2 values, using a normalisation factor computed from the WT cells on each plate

    To be more specific, we want to remove inter-plate variation in fv_fm values, so we compute the average fv_fm
    from the WT cells on each plate, and then compute the mean of these values (so each plate is weighted equally).
    Then, the fv_fm values and Y(II) values are multiplied by a scale factor so the average fv_fm of all wild type
    cells is equal between plates.
    """
    df = get_y2_df()
    df = df[(df["light_regime"] == "20h_HL") | (df["light_regime"] == "20h_ML")]
    df_wt = df[df["mutant_ID"] == "WT"]

    fvfm_means = df_wt.groupby(["plate", "measurement"])["fv_fm"].mean()
    norm_value = fvfm_means.mean()
    scale_facs = norm_value / fvfm_means

    assert scale_facs.min() > 0.9
    assert scale_facs.max() < 1.1

    # Merge back with original df
    df = df.merge(scale_facs, on=["plate", "measurement"], suffixes=("", "_scale_fac"), validate="m:1")

    # Multiply each y2 column by the scale factor and store in new dataframe
    cols = [f"y2_{i}" for i in range(1, 82)] + ["fv_fm"]
    df_scaled = df.copy()
    for col in cols:
        df_scaled[col] = df_scaled[col] * df_scaled[f"fv_fm_scale_fac"]

    # Check that the averaged fv_fm values are now all equal to the normalisation value
    fvfm_means = df_scaled[df_scaled["mutant_ID"] == "WT"].groupby(["plate", "measurement"])["fv_fm"].mean()
    assert abs(fvfm_means - norm_value).max() < 1e-6

    return df_scaled


def main():
    # If this script is run as a main script, then plot information on differences between WT Y(II) values to understand
    # the effect of normalising using these values

    # Do not truncate pandas output
    pd.set_option('display.max_columns', None)

    df = get_y2_df()

    # Print all unique (plate, measurement) pairs and corresponding number of rows
    print(df.groupby(["plate", "measurement"]).size())

    df = df[(df["light_regime"] == "20h_HL") | (df["light_regime"] == "20h_ML")]
    df_wt = df[df["mutant_ID"] == "WT"]
    df_non_wt = df[df["mutant_ID"] != "WT"]

    print(df.head())

    # Plot the average fv_fm for each plate, measurement pair, for WT and non-WT cells
    plot_avg_fv_fm(df_non_wt, df_wt)

    # We want to compute an average fv_fm between WTs on each plate

    # Plot the distribution of average values first, should be gaussian from central limit theorem
    plot_fv_fm_distribution(df_wt)

    df_scaled = get_normalised_y2_df()

    # Plot the Y(II) time series in various ways
    plot_rescaled_data_means(df_scaled)
    plot_rescaled_data_all(df_scaled)
    plot_rescaled_data_means(df)



def plot_rescaled_data_all(df_scaled):
    # Plot the Y(II) time series by plate, separately for WT and non-WT cells
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "Computer Modern Serif"
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), sharey=True, sharex=True)
    df1 = df_scaled[(df_scaled["mutant_ID"] == "WT") & (df_scaled["light_regime"] == "20h_HL")]
    df2 = df_scaled[(df_scaled["mutant_ID"] != "WT") & (df_scaled["light_regime"] == "20h_HL")]
    df3 = df_scaled[(df_scaled["mutant_ID"] == "WT") & (df_scaled["light_regime"] == "20h_ML")]
    df4 = df_scaled[(df_scaled["mutant_ID"] != "WT") & (df_scaled["light_regime"] == "20h_ML")]
    cols = ["fv_fm"] + [f"y2_{i}" for i in range(1, 82)]
    xrange = range(82)

    # Create colormap for plates
    cmap = plt.cm.get_cmap("tab20")
    plates = df1["plate"].unique()
    plate_colors = {plate: cmap(i) for i, plate in enumerate(plates)}

    ax = axs[0, 0]
    ax.set_title("WT 20h HL")
    ax.set_ylabel("Y(II)")
    ax.set_ylim(-0.1, 0.8)
    for i, row in list(df1.iterrows())[::-1]:
        ax.plot(xrange, row[cols], color=plate_colors[row["plate"]], alpha=0.2)

    ax = axs[0, 1]
    ax.set_title("Mutant 20h HL")
    for i, row in df2.iterrows():
        ax.plot(xrange, row[cols], color=plate_colors[row["plate"]], alpha=0.1)

    ax = axs[1, 0]
    ax.set_ylabel("Y(II)")
    ax.set_xlabel("Time point")
    ax.set_title("WT 20h ML")
    for i, row in list(df3.iterrows())[::-1]:
        ax.plot(xrange, row[cols], color=plate_colors[row["plate"]], alpha=0.2)

    ax = axs[1, 1]
    ax.set_xlabel("Time point")
    ax.set_title("Mutant 20h ML")
    for i, row in df4.iterrows():
        ax.plot(xrange, row[cols], color=plate_colors[row["plate"]], alpha=0.1)

    # Add legend for plates
    handles = [plt.Line2D([0, 0], [0, 0], color=plate_colors[i], marker='o', linestyle='') for i in plates]
    labels = [f"Plate {i}" for i in plates]

    # Remove all top and right borders
    for ax in axs.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    # Place legend on right hand side in the middle, vertically centred
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.0, 0.5))
    plt.show()


def plot_rescaled_data_means(df_scaled):
    # Plot the mean Y(II) by plate, separately for WT and non-WT cells
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "Computer Modern Serif"
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), sharey=True, sharex=True)
    df1 = df_scaled[(df_scaled["mutant_ID"] == "WT") & (df_scaled["light_regime"] == "20h_HL")]
    df2 = df_scaled[(df_scaled["mutant_ID"] != "WT") & (df_scaled["light_regime"] == "20h_HL")]
    df3 = df_scaled[(df_scaled["mutant_ID"] == "WT") & (df_scaled["light_regime"] == "20h_ML")]
    df4 = df_scaled[(df_scaled["mutant_ID"] != "WT") & (df_scaled["light_regime"] == "20h_ML")]
    cols = ["fv_fm"] + [f"y2_{i}" for i in range(1, 82)]
    xrange = range(82)

    # Create colormap for plates
    cmap = plt.cm.get_cmap("tab20")
    plates = df1["plate"].unique()
    plate_colors = {plate: cmap(i) for i, plate in enumerate(plates)}

    ax = axs[0, 0]
    ax.set_title("WT 20h HL")
    ax.set_ylabel("Y(II)")
    ax.set_ylim(0, 0.7)
    plate_means = df1.groupby("plate")[cols].mean()
    for i, row in plate_means.iterrows():
        ax.plot(xrange, row[cols], color=plate_colors[i], alpha=1.0)

    ax = axs[0, 1]
    ax.set_title("Mutant 20h HL")
    plate_means = df2.groupby("plate")[cols].mean()
    for i, row in plate_means.iterrows():
        ax.plot(xrange, row[cols], color=plate_colors[i], alpha=1)

    ax = axs[1, 0]
    ax.set_ylabel("Y(II)")
    ax.set_xlabel("Time point")
    ax.set_title("WT 20h ML")
    plate_means = df3.groupby("plate")[cols].mean()
    for i, row in plate_means.iterrows():
        ax.plot(xrange, row[cols], color=plate_colors[i], alpha=1)

    ax = axs[1, 1]
    ax.set_xlabel("Time point")
    ax.set_title("Mutant 20h ML")
    plate_means = df4.groupby("plate")[cols].mean()
    for i, row in plate_means.iterrows():
        ax.plot(xrange, row[cols], color=plate_colors[i], alpha=1)

    # Add legend for plates
    handles = [plt.Line2D([0, 0], [0, 0], color=plate_colors[i], marker='o', linestyle='') for i in plates]
    labels = [f"Plate {i}" for i in plates]

    # Remove all top and right borders
    for ax in axs.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    # Place legend on right hand side in the middle, vertically centred
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.0, 0.5))
    plt.show()


def plot_fv_fm_distribution(df_wt):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "Computer Modern Serif"
    fvfm_means = df_wt.groupby(["plate", "measurement"])["fv_fm"].mean().plot.hist(bins=10, density=True)
    fvfm_means = df_wt.groupby(["plate", "measurement"])["fv_fm"].mean().plot.kde()
    plt.xlabel("Fv/Fm")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()


def plot_avg_fv_fm(df_non_wt, df_wt):
    # Sort by measurement_time_0
    #df = df.sort_values(by="measurement_time_0")

    # From each plate, measurement pair plot the fv_fm values for the WT cells
    plt.rcParams.update({'text.usetex': True})
    plt.rcParams["font.family"] = "Computer Modern Serif"
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    ax = axs[0]
    df_wt.groupby(["plate", "measurement"])["fv_fm"].mean().plot(kind="bar", ax=ax,
                                                                 yerr=df_wt.groupby(["plate", "measurement"])[
                                                                     "fv_fm"].std())
    ax.set_ylim(0.53, 0.68)
    ax.set_ylabel("Fv/Fm")
    ax.set_title("WT Fv/Fm")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax = axs[1]
    df_non_wt.groupby(["plate", "measurement"])["fv_fm"].mean().plot(kind="bar", ax=ax,
                                                                     yerr=df_non_wt.groupby(["plate", "measurement"])[
                                                                         "fv_fm"].std())
    ax.set_ylim(0.53, 0.68)
    ax.set_ylabel("Fv/Fm")
    ax.set_title("Mutant Fv/Fm")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()