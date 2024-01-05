"""In this file, we take preprocessed image data (segmented into wells) and construct a database (currently csv files and sqlite)
containing all the features we need for our analysis.

The script is controlled using hard-coded constants at the top of the file. These are:
    - DEV_MODE: whether to run in development mode (only use a few files)
    - INPUT_DIR: directory containing the preprocessed image data
    - IDENTITY_SPREADSHEET_PATH: path to the spreadsheet containing information about the identity of each well.
        This is found at: https://docs.google.com/spreadsheets/d/1_UcLC4jbI04Rnpt2vUkSCObX8oUY6mzl/edit#gid=206647583
    - OUTPUT_DIR: directory to write the csv files to

The database will have five tables:
    - plate_info: contains information about each plate, such as plate number, light regime, etc.
    - image_features: contains features extracted from the images, such as Fv/Fm, Y2, NPQ, along with experimental information
    - identity: contains information about the identity of each mutant, such as well location, plate number, etc.
    - mutations: contains information about the mutations in each mutant, such as disrupted gene name, type, confidence level, etc.
    - gene_descriptions: contains lengthy descriptions of each gene
"""
import datetime
from itertools import product
from pathlib import Path
import logging
import sqlite3

import pandas as pd
import numpy as np

from chlamy_impi.database_creation.error_correction import fix_erroneous_time_points, remove_repeated_initial_frame
from chlamy_impi.database_creation.utils import location_to_index, parse_name, spreadsheet_plate_to_numeric, \
    compute_measurement_times
from chlamy_impi.lib.fv_fm_functions import compute_all_fv_fm_averaged
from chlamy_impi.lib.mask_functions import compute_threshold_mask
from chlamy_impi.lib.npq_functions import compute_all_npq_averaged
from chlamy_impi.lib.y2_functions import compute_all_y2_averaged

logger = logging.getLogger(__name__)

DEV_MODE = False

INPUT_DIR = Path("../../data")
IDENTITY_SPREADSHEET_PATH = Path(
    "../../data/Identity plates in Burlacot Lab 20231221 simplified.xlsx - large-lib_rearray2.txt.csv")
OUTPUT_DIR = Path("./../../output/database_creation/v2")


def construct_plate_info_dataframe():
    """In this function, we construct a dataframe containing information about each plate

    This includes:
        - Plate number
        - Start date
        - Light regime
        - Threshold
        - Number of frames
        - Measurement times


    TODO: add remaining experimental columns:
     - Was there an issue?
     - Temperature under camera (avg, max, min)
     - Temperature in algae house
     - # days M plate grown
     - # days S plate grown
     - Time duration of experiment corresponding to each time point
    """
    filenames_meta, filenames_npy = get_filenames()

    rows = []

    for filename_npy, filename_meta in zip(filenames_npy, filenames_meta):
        logger.info(f"Processing plate info from filename: {filename_npy.name}")
        plate_num, measurement_num, light_regime = parse_name(filename_npy)

        img_array, meta_df = prepare_img_array_and_df(filename_meta, filename_npy)

        measurement_times = compute_measurement_times(meta_df=meta_df)

        _, threshold = compute_threshold_mask(img_array, return_threshold=True)

        rows.append({
            "plate": plate_num,
            "measurement": measurement_num,
            "light_regime": light_regime,
            "threshold": threshold,
            "num_frames": img_array.shape[2],
        })

        for i in range(82):
            try:
                rows[-1][f"measurement_time_{i}"] = measurement_times[i]
            except IndexError:
                rows[-1][f"measurement_time_{i}"] = np.nan

    df = pd.DataFrame(rows)
    logger.info(f"Constructed plate info dataframe. Shape: {df.shape}. Columns: {df.columns}.")
    logger.info(f"{df.head()}")
    return df


def prepare_img_array_and_df(filename_meta, filename_npy):
    img_array = np.load(filename_npy)
    img_array = remove_repeated_initial_frame(img_array)
    meta_df = pd.read_csv(filename_meta, header=0, delimiter=";").iloc[:, :-1]
    meta_df, img_array = fix_erroneous_time_points(meta_df, img_array)
    return img_array, meta_df


def get_filenames():
    assert INPUT_DIR.exists()

    filenames_npy = list(INPUT_DIR.glob("*.npy"))
    filenames_npy.sort()

    filenames_meta = [x.with_suffix(".csv") for x in filenames_npy]

    if DEV_MODE:
        filenames_npy = filenames_npy[:10]
        filenames_meta = filenames_meta[:10]
        logger.info(f"DEV_MODE: only using {len(filenames_meta)} files")

    # Check that these two lists of filenames are the same
    assert len(filenames_npy) == len(filenames_meta)
    for f1, f2 in zip(filenames_npy, filenames_meta):
        assert f1.stem == f2.stem, f"{f1.stem} != {f2.stem}"
        assert f2.exists(), f"{f2} does not exist"

    logger.info(f"Found {len(filenames_npy)} files in {INPUT_DIR}")

    return filenames_meta, filenames_npy


def construct_img_feature_dataframe() -> pd.DataFrame:
    """In this function, we extract all image features, such as Fv/Fm, Y2, NPQ

    TODO:
     - Other quantifiers of fluorescence or shape heterogeneity
    """
    filenames_meta, filenames_npy = get_filenames()

    rows = []

    for filename_npy, filename_meta in zip(filenames_npy, filenames_meta):
        plate_num, measurement_num, light_regime = parse_name(filename_npy)

        logger.info(f"\n\n\nProcessing image features from filename: {filename_npy.name}")

        img_array, meta_df = prepare_img_array_and_df(filename_meta, filename_npy)

        assert img_array.shape[2] % 2 == 0

        mask_array = compute_threshold_mask(img_array, return_threshold=False)
        fv_fm_array = compute_all_fv_fm_averaged(img_array, mask_array)
        y2_array = compute_all_y2_averaged(img_array, mask_array)

        npq_array = compute_all_npq_averaged(img_array, mask_array)

        Ni, Nj = img_array.shape[:2]

        for i, j in product(range(Ni), range(Nj)):
            row_data = {
                "plate": plate_num,
                "measurement": measurement_num,
                "i": i,
                "j": j,
                "fv_fm": fv_fm_array[i, j],
                "mask_area": np.sum(mask_array[i, j]),
            }

            assert len(y2_array[i, j]) <= 81
            assert len(npq_array[i, j]) <= 81
            assert len(y2_array[i, j]) == len(npq_array[i, j])

            for tstep in range(1, 82):  # There can be at most 81 time steps (82 pairs, but the first is fv/fm)
                try:
                    row_data[f"y2_{tstep}"] = y2_array[i, j, tstep - 1]
                except IndexError:
                    row_data[f"y2_{tstep}"] = np.nan

            for tstep in range(1, 82):
                try:
                    row_data[f"npq_{tstep}"] = npq_array[i, j, tstep - 1]
                except IndexError:
                    row_data[f"npq_{tstep}"] = np.nan

            rows.append(row_data)

    df = pd.DataFrame(rows)

    logger.info(f"Constructed image features dataframe. Shape: {df.shape}. Columns: {df.columns}.")
    logger.info(f"{df.head()}")
    return df


def construct_description_dataframe() -> pd.DataFrame:
    """In this function, we extract all gene descriptions, and store as a separate dataframe

    Each gene has one description, but the descriptions are very long, so we store them separately
    """
    assert IDENTITY_SPREADSHEET_PATH.exists()
    df = pd.read_csv(IDENTITY_SPREADSHEET_PATH, header=0)

    # Create new dataframe with just the gene descriptions, one for each gene
    df_gene_descriptions = df[["gene", "description"]]
    df_gene_descriptions = df_gene_descriptions.drop_duplicates(subset=["gene"])

    logger.info(f"Constructed description dataframe. Shape: {df_gene_descriptions.shape}.")
    logger.info(f"{df_gene_descriptions.head()}")
    return df_gene_descriptions


def construct_mutations_dataframe() -> pd.DataFrame:
    """In this function, we extract all mutation features, such as gene name, location, etc.

    This is a separate table because each mutant_ID can have several mutations
    """
    df = pd.read_csv(IDENTITY_SPREADSHEET_PATH, header=0)

    df = df[["mutant_ID", "gene", "feature", "confidence_level"]]
    df = df.drop_duplicates(ignore_index=True)

    logger.info(f"Constructed mutation dataframe. Shape: {df.shape}. Columns: {df.columns}.")
    logger.info(f"{df.head()}")
    return df


def construct_identity_dataframe(mutation_df: pd.DataFrame) -> pd.DataFrame:
    """In this function, we extract all identity features, such as strain name, location, etc.

    There is a single row per mutant_ID
    """
    df = pd.read_csv(IDENTITY_SPREADSHEET_PATH, header=0)

    # Assert that there are no null values in the "Location" and "New Location" columns
    assert df["Location"].notnull().all()
    assert df["New location"].notnull().all()

    # Collect columns which we need
    df = df.rename(columns={"New location": "plate", "Location": "location"})
    df = df[["mutant_ID", "plate", "location"]]
    df = df.drop_duplicates(ignore_index=True)

    # Add columns to locate each well (i is row #, j is column #)
    df["plate"] = df["plate"].apply(spreadsheet_plate_to_numeric)
    df["i"] = df["location"].apply(lambda x: location_to_index(x)[0])
    df["j"] = df["location"].apply(lambda x: location_to_index(x)[1])

    # Add column which tells us the number of genes which were mutated
    gene_mutation_counts = mutation_df.groupby("mutant_ID").count()["gene"]
    df["num_mutations"] = df["mutant_ID"].apply(lambda x: gene_mutation_counts[x])

    logger.info(f"Constructed identity dataframe. Shape: {df.shape}. Columns: {df.columns}.")
    return df


def write_dataframe(df: pd.DataFrame, name: str):
    """In this function, we write the dataframe to a csv file
    """
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)
    df.to_csv(OUTPUT_DIR / name)

    logger.info(f"Written dataframe to {OUTPUT_DIR / name}")


def create_sqlite_database(name_to_df: dict[str, pd.DataFrame]):
    """In this function, we create a sqlite database from the two dataframes
    """
    conn = sqlite3.connect(OUTPUT_DIR / "chlamy_screen_database.db")
    for name, df in name_to_df.items():
        df.to_sql(name, conn, if_exists="replace")
    conn.close()


def main():
    """This is the main function, showing the high level approach to constructing the database
    """
    plate_info_df = construct_plate_info_dataframe()
    image_features_df = construct_img_feature_dataframe()
    descriptions_df = construct_description_dataframe()
    mutations_df = construct_mutations_dataframe()
    identity_df = construct_identity_dataframe(mutations_df)

    name_to_df = {
        "plate_info": plate_info_df,
        "image_features": image_features_df,
        "identity": identity_df,
        "gene_descriptions": descriptions_df,
        "mutations": mutations_df,
    }

    for name, df in name_to_df.items():
        write_dataframe(df, f"{name}.csv")

    create_sqlite_database(name_to_df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG if DEV_MODE else logging.INFO)
    main()
