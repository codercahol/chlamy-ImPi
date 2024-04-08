"""In this file, we take preprocessed image data (segmented into wells) and construct a database (currently csv files and sqlite)
containing all the features we need for our analysis.

The script is controlled using hard-coded constants at the top of the file. These are:
    - DEV_MODE: whether to run in development mode (only use a few files)

The database will have five tables:
    - plate_info: contains information about each plate, such as plate number, light regime, etc.
    - image_features: contains features extracted from the images, such as Fv/Fm, Y2, NPQ, along with experimental information
    - identity: contains information about the identity of each mutant, such as well location, plate number, etc.
    - mutations: contains information about the mutations in each mutant, such as disrupted gene name, type, confidence level, etc.
    - gene_descriptions: contains lengthy descriptions of each gene
"""

from itertools import product
import logging
import re

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from chlamy_impi.database_creation.error_correction import (
    remove_repeated_initial_frame, manually_fix_erroneous_time_points,
)
from chlamy_impi.paths import get_npy_and_csv_filenames


def investigate_erroneous_time_points(meta_df):
    # Print out the all rows of meta_df which have any zeros
    zeros = False

    df = meta_df.iloc[:, 4:]
    cols = df.columns

    for i, row in df.iterrows():
        for col in cols:
            if row[col] == 0.0:
                print(f"Row {i} has zeros in {col}: {row}")
                zeros = True

    return zeros


def main():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    filenames_meta, filenames_npy = get_npy_and_csv_filenames()

    zeros_arr = []
    frames_arr = {}

    for filename_npy, filename_meta in zip(filenames_npy, filenames_meta):
        print()
        print(filename_npy.stem)

        meta_df = pd.read_csv(filename_meta, header=0, delimiter=";").iloc[:, :-1]
        img_array = np.load(filename_npy)
        img_array = remove_repeated_initial_frame(img_array)

        try:
            meta_df, img_array = manually_fix_erroneous_time_points(meta_df, img_array, filename_npy.stem)
        except AssertionError as e:
            print(e)
            pass

        num_frames = img_array.shape[2]
        if not ((num_frames == 84) or (num_frames == 164)):
            print(str(filename_npy))
            print(f"Number of frames in {filename_npy} is {num_frames}")

            zeros = investigate_erroneous_time_points(meta_df)
            zeros_arr.append(zeros)

            frames_arr[str(filename_npy)] = num_frames

            # Use this plot to cross-check whether a frame really is erroneous
            avg_intensities = np.mean(img_array, axis=(0, 1, 3, 4))
            plt.plot(avg_intensities)
            plt.title(filename_npy.stem)
            plt.show()

    print()
    print('filename to unexpected frame number', frames_arr)



if __name__ == "__main__":
    main()
