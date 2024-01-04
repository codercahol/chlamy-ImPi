import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_csvs():
    outdir = Path("./../../output/database_creation/v1")

    name_to_df = {}

    for filename in outdir.glob("*.csv"):
        df = pd.read_csv(filename, header=0)

        name_to_df[filename.stem] = df

    return name_to_df


def main():
    name_to_df = load_csvs()

    for name, df in name_to_df.items():
        print(name)
        print(df.head())
        print()

    # Extract 5-M6 rows from the image feature df
    df = name_to_df["image_features"]
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
