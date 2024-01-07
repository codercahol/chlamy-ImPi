import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pyarrow.parquet as pq


def read_df_from_parquet(columns: list[str] = None
                         ) -> pd.DataFrame:
    filename = "./../../output/database_creation/database.parquet"
    table = pq.read_table(filename, columns=columns)
    df = table.to_pandas()
    return df


def plot_all_time_series(df):
    # Find all plates and M#s from df_plate first
    plates = df["plate"].unique()
    measurements = df["measurement"].unique()

    print(plates)
    print(measurements)
    assert 0

    # For each plate and measurement, plot all time series if the data exists
    for plate in plates:
        for m in measurements:
            df_subset = df[(df["plate"] == plate) & (df["measurement"] == m)]

            if len(df_subset) > 0:
                light_treatment = \
                    df[(df["plate"] == plate) & (df["measurement"] == m)]["light_regime"].values[0]

                time_series_filename = Path("./../../output/data_exploration/v1") / f"npq_and_y2_timeseries_{plate}_{m}.png"
                plot_both_time_series(df_subset, plate, m, light_treatment, time_series_filename)
                print(f"Plot written to {time_series_filename}")

                time_series_filename = Path(
                    "./../../output/data_exploration/v1") / f"y2_timeseries_{plate}_{m}.png"
                plot_y2_time_series(df_subset, plate, m, light_treatment, time_series_filename)
                print(f"Plot written to {time_series_filename}")

            else:
                print(f"No data for plate {plate} measurement {m}")


def plot_y2_time_series(df, plate, m, treatment, filename):
    y2_cols = [f"y2_{i}" for i in range(1, 82)]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for i, row in df.iterrows():
        y2_vals = row[y2_cols].values
        ax.plot(y2_vals, color="black", alpha=0.2)

    ax.set_ylim(-0.2, 1)
    ax.set_xlabel("Time point")
    ax.set_ylabel("Y(II)")
    ax.set_title(f"Y(II) time series for plate {plate} measurement {m}: {treatment}")

    # Only have bounding box on bottom left
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(filename)
    plt.close(fig)


def plot_both_time_series(df, plate, m, treatment, filename):
    y2_cols = [f"y2_{i}" for i in range(1, 82)]
    npq_cols = [f"npq_{i}" for i in range(1, 82)]

    fig, axs = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    for i, row in df.iterrows():
        y2_vals = row[y2_cols].values
        npq_vals = row[npq_cols].values
        axs[0].plot(y2_vals, color="black", alpha=0.2)
        axs[1].plot(npq_vals, color="black", alpha=0.2)

    #axs[0].set_xlabel("Time point")
    axs[0].set_ylabel("Y2")
    axs[0].set_title(f"Y2 time series for plate {plate} measurement {m}: {treatment}")

    axs[1].set_xlabel("Time point")
    axs[1].set_ylabel("NPQ")
    axs[1].set_title(f"NPQ time series for plate {plate} measurement {m}: {treatment}")

    fig.savefig(filename)
    plt.close(fig)



def main():
    df = read_df_from_parquet()
    print(df.head())

    plot_all_time_series(df)

    assert 0


    df = df[(df["plate"] == 5) & (df["measurement"] == "M6")]
    print(df)

    # Extract time time series of Y2 for each well
    y2_cols = [f"y2_{i}" for i in range(1, 82)]
    npq_cols = [f"npq_{i}" for i in range(1, 82)]
    y2_df = df[["i", "j"] + y2_cols]
    npq_df = df[["i", "j"] + npq_cols]
    fv_fm_df = df[["i", "j"] + ["fv_fm"]]
    fv_fm_df["ij"] = fv_fm_df["i"].astype(str) + "-" + fv_fm_df["j"].astype(str)
    fv_fm_df = fv_fm_df.drop(columns=["i", "j"])
    fv_fm_df = fv_fm_df.set_index("ij")

    print(y2_df.head())

    # Plot Y2 for each well over time, each well is given by i and j
    # Combine i and j into single column
    y2_df["ij"] = y2_df["i"].astype(str) + "-" + y2_df["j"].astype(str)
    y2_df = y2_df.drop(columns=["i", "j"])
    y2_df = y2_df.set_index("ij")

    # Wild type locations (i, j) are: (2, 11), (13, 2), (13, 21)
    # Add column to check if location is wild type
    y2_df["wild_type"] = y2_df.index.isin(["2-11", "13-2", "13-21"])

    #plot_y2_timeseries(y2_cols, y2_df)
    #plot_histogram_of_y2_averages(y2_df)
    # Fit trend line to each time series and plot histogram of slopes


    # Repeat above but for npq
    npq_df["ij"] = npq_df["i"].astype(str) + "-" + npq_df["j"].astype(str)
    npq_df = npq_df.drop(columns=["i", "j"])
    npq_df = npq_df.set_index("ij")

    # Wild type locations (i, j) are: (2, 11), (13, 2), (13, 21)
    # Add column to check if location is wild type
    npq_df["wild_type"] = npq_df.index.isin(["2-11", "13-2", "13-21"])

    #plot_histogram_of_npq_averages(npq_df)


    # Scatter plot og Y2 average vs NPQ average
    # Scatter plot of Y2 average vs NPQ average, color by wild type or not
    # First merge y2 and npq dataframes on ij, only keep average column
    y2_df["average"] = y2_df.mean(axis=1)
    npq_df["average"] = npq_df.mean(axis=1)
    merged_df = y2_df[["average", "wild_type"]].merge(npq_df[["average"]], on="ij", suffixes=("_y2", "_npq"))

    print(merged_df)
    #plot_scatter_of_averages(merged_df)

    # Extract Y2 and NPQ for all odd numbered time points, these are light measurements
    y2_light_cols = [f"y2_{i}" for i in range(1, 82, 2)]
    npq_light_cols = [f"npq_{i}" for i in range(1, 82, 2)]

    # Repeat the scatter plot, only using light measurements
    y2_df_light = y2_df[["wild_type"] + y2_light_cols]
    npq_df_light = npq_df[npq_light_cols]
    y2_df_light["average"] = y2_df_light.mean(axis=1)
    npq_df_light["average"] = npq_df_light.mean(axis=1)
    merged_df_light = y2_df_light[["average", "wild_type"]].merge(npq_df_light[["average"]], on="ij", suffixes=("_y2", "_npq"))
    merged_df_light = merged_df_light.merge(fv_fm_df, on="ij")

    #plot_scatter_of_averages(merged_df_light)
    plot_fv_fm_vs_y2_scatter(merged_df_light)


def plot_fv_fm_vs_y2_scatter(merged_df):
    # Skip rows with nan fv_fm
    merged_df = merged_df[~merged_df["fv_fm"].isna()]

    # Set colormap to viridis and add colorbar
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.scatter(merged_df["average_y2"], merged_df["fv_fm"])

    # Fit linear regression line and display equation
    m, b = np.polyfit(merged_df["average_y2"].astype(float), merged_df["fv_fm"].astype(float), 1)
    ax.plot(merged_df["average_y2"], m * merged_df["average_y2"] + b, color="black")
    ax.text(0.5, 0.1, f"fv/fm = {m:.3f}Y2 + {b:.3f}", transform=ax.transAxes)

    wt_df = merged_df[merged_df["wild_type"]]
    ax.scatter(wt_df["average_y2"], wt_df["fv_fm"], color="red", marker="o", facecolors="none", s=80)
    ax.set_xlabel("Average Y2")
    ax.set_ylabel("fv/fm")
    ax.set_title("Average Y2 vs fv/fm")
    plt.show()


def plot_scatter_of_averages(merged_df):
    # Skip rows with nan fv_fm
    merged_df = merged_df[~merged_df["fv_fm"].isna()]

    # Set colormap to viridis and add colorbar
    fig, ax = plt.subplots(figsize=(20, 10))
    im = ax.scatter(merged_df["average_y2"], merged_df["average_npq"], c=merged_df["fv_fm"], cmap="viridis", s=40)
    fig.colorbar(im, ax=ax)

    wt_df = merged_df[merged_df["wild_type"]]
    ax.scatter(wt_df["average_y2"], wt_df["average_npq"], color="red", marker="o", facecolors="none", s=80)
    ax.set_xlabel("Average Y2")
    ax.set_ylabel("Average NPQ")
    ax.set_title("Average Y2 vs Average NPQ")
    plt.show()


def plot_histogram_of_y2_averages(y2_df):
    y2_df["average"] = y2_df.mean(axis=1)

    # Wild type average Y2 values
    wild_type_averages = y2_df[y2_df["wild_type"]]["average"].values

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.hist(y2_df["average"], bins=100)
    ax.axvline(x=wild_type_averages.mean(), color="red", label="Wild type average")
    ax.set_xlabel("Average Y2")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Average Y2 Values")
    plt.show()


def plot_histogram_of_npq_averages(npq_df):
    npq_df["average"] = npq_df.mean(axis=1)

    # Wild type average NPQ values
    wild_type_averages = npq_df[npq_df["wild_type"]]["average"].values

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.hist(npq_df["average"], bins=100)
    ax.axvline(x=wild_type_averages.mean(), color="red", label="Wild type average")
    ax.set_xlabel("Average NPQ")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Average NPQ Values")
    plt.show()



def plot_y2_timeseries(y2_cols, y2_df):
    fig, ax = plt.subplots(figsize=(20, 10))
    for i, row in y2_df.iterrows():
        y2_vals = row[y2_cols].values
        ax.plot(y2_vals)
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
