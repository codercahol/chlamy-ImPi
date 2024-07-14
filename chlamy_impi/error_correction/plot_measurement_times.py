from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from chlamy_impi.database_creation.error_correction import remove_repeated_initial_frame
from chlamy_impi.database_creation.utils import parse_name
from chlamy_impi.paths import get_npy_and_csv_filenames


def combine_date_and_time(dates: list[str], times: list[str]) -> list[pd.Timestamp]:
    """
    Combines two lists of date and time strings into a single list of datetime objects.

    Parameters:
    dates (list of str): List of date strings in "DD.MM.YY" format.
    times (list of str): List of time strings in "HH:MM:SS" format.

    Returns:
    list of pd.Timestamp: List of datetime objects corresponding to the given dates and times.
    """
    if len(dates) != len(times):
        raise ValueError("The lengths of dates and times lists must be the same.")

    # List to store combined datetime strings
    datetime_strings = [f"{date} {time}" for date, time in zip(dates, times)]

    # Convert the list of datetime strings to datetime objects
    datetimes = pd.to_datetime(datetime_strings, format="%d.%m.%y %H:%M:%S")

    return datetimes.values


def main():
    filenames_meta, filenames_npy = get_npy_and_csv_filenames()

    time_regime_to_metadf = defaultdict(list)

    for filename_npy, filename_meta in zip(filenames_npy, filenames_meta):
        plate_num, measurement_num, time_regime = parse_name(filename_npy.name)

        meta_df = pd.read_csv(filename_meta, header=0, delimiter=";").iloc[:, :-1]

        img_array = np.load(filename_npy)
        img_array_new = remove_repeated_initial_frame(img_array)
        assert img_array_new.shape == img_array.shape

        time_regime_to_metadf[time_regime].append((plate_num, meta_df))

    print(time_regime_to_metadf.keys())

    # Now we want to look at the relative measurement times for each measurement_num

    for time_regime, data in time_regime_to_metadf.items():

        fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=200)

        for plate_num, df in data:
            times = df["Time"].values
            dates = df["Date"].values
            timestamps = combine_date_and_time(dates, times)
            reference_time = timestamps[0]
            relative_time_hrs = (timestamps - reference_time).astype('timedelta64[s]') / 3600.0
            axs[0].scatter(range(len(relative_time_hrs)), relative_time_hrs, label=f"Plate {plate_num}", s=4)

            time_increment_mins = np.diff(timestamps).astype('timedelta64[s]') / 60.0
            axs[1].scatter(range(len(time_increment_mins)), time_increment_mins, label=f"Plate {plate_num}", s=4)

        ax = axs[0]
        ax.set_title(f"Time regime: {time_regime}")
        ax.set_ylabel("Time (hours)")
        ax.set_xlabel("Frame number")

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        ax = axs[1]
        ax.set_title(f"Time regime: {time_regime}")
        ax.set_ylabel("Time increment relative to last measurement (mins)")
        ax.set_xlabel("Frame number")

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        fig.savefig(f"time_regime_{time_regime}.png")
        plt.close()






if __name__ == '__main__':
    main()
