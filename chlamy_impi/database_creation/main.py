from itertools import product
from pathlib import Path
import logging
import sqlite3

import pandas as pd
import numpy as np

from chlamy_impi.database_creation.utils import location_to_index, parse_name, spreadsheet_plate_to_numeric
from chlamy_impi.lib.fv_fm_functions import compute_all_fv_fm_averaged
from chlamy_impi.lib.mask_functions import compute_threshold_mask
from chlamy_impi.lib.npq_functions import compute_all_npq_averaged
from chlamy_impi.lib.y2_functions import compute_all_y2_averaged

logger = logging.getLogger(__name__)

DEV_MODE = True
INPUT_DIR = Path("../../output/image_processing/v6/img_array")
IDENTITY_SPREADSHEET_PATH = Path(
    "../../data/Identity plates in Burlacot Lab 20231221 simplified.xlsx - large-lib_rearray2.txt.csv")
OUTPUT_DIR = Path("./../../output/database_creation/v1")


def construct_img_feature_dataframe():
    """In this function, we extract all image features, such as Fv/Fm, Y2, NPQ
    """
    assert INPUT_DIR.exists()
    filenames = list(INPUT_DIR.glob("*.npy"))

    if DEV_MODE:
        filenames = filenames[:1]
        logger.info(f"DEV_MODE: only using {len(filenames)} files")

    rows = []

    for filename in filenames:
        plate_num, measurement_num = parse_name(filename)

        logger.info(f"Processing image features from filename: {filename.name}")

        img_array = np.load(filename)
        mask_array = compute_threshold_mask(img_array)
        fv_fm_array = compute_all_fv_fm_averaged(img_array, mask_array)
        y2_array = compute_all_y2_averaged(img_array, mask_array)
        npq_array = compute_all_npq_averaged(img_array, mask_array)

        # Pad time series to have 168 entries (which is the maximum)
        Ni, Nj = img_array.shape[:2]

        for i, j in product(range(Ni), range(Nj)):
            row_data = {
                "plate": plate_num,
                "measurement": measurement_num,
                "i": i,
                "j": j,
                "fv_fm": fv_fm_array[i, j],
            }

            for tstep in range(84):
                try:
                    row_data[f"y2_{tstep}"] = y2_array[i, j, tstep]
                    row_data[f"npq_{tstep}"] = npq_array[i, j, tstep]
                except IndexError:
                    row_data[f"y2_{tstep}"] = np.nan
                    row_data[f"npq_{tstep}"] = np.nan

            rows.append(row_data)

    df = pd.DataFrame(rows)
    df = df.set_index(["plate", "i", "j"])

    logger.info(f"Constructed dataframe. Shape: {df.shape}. Columns: {df.columns}.")
    return df


def construct_identity_dataframe() -> pd.DataFrame:
    """In this function, we extract all identity features, such as strain name, location, etc.
    """
    df = pd.read_csv(IDENTITY_SPREADSHEET_PATH, header=0)

    # Assert that there are no null values in the "Location" and "New Location" columns
    assert df["Location"].notnull().all()
    assert df["New location"].notnull().all()

    # Collect columns which we need
    df = df.rename(columns={"New location": "plate", "Location": "location"})
    df = df[["plate", "location", "mutant_ID", "gene", "feature", "confidence_level", "description"]]
    df["plate"] = df["plate"].apply(spreadsheet_plate_to_numeric)
    df["i"] = df["location"].apply(lambda x: location_to_index(x)[0])
    df["j"] = df["location"].apply(lambda x: location_to_index(x)[1])
    df = df.set_index(["plate", "i", "j"])

    logger.info(f"Constructed dataframe. Shape: {df.shape}. Columns: {df.columns}.")
    return df


def write_dataframe(df: pd.DataFrame, name: str):
    """In this function, we write the dataframe to a csv file
    """
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)
    df.to_csv(OUTPUT_DIR / name)

    logger.info(f"Written dataframe to {OUTPUT_DIR / name}")


def create_sqlite_database(image_features_df: pd.DataFrame, identity_df: pd.DataFrame):
    """In this function, we create a sqlite database from the two dataframes
    """
    conn = sqlite3.connect(OUTPUT_DIR / "chlamy_screen_database.db")
    image_features_df.to_sql("image_features", conn)
    identity_df.to_sql("identity", conn)
    conn.close()


def main():
    image_features_df = construct_img_feature_dataframe()
    identity_df = construct_identity_dataframe()

    write_dataframe(image_features_df, "image_features.csv")
    write_dataframe(identity_df, "identity.csv")

    create_sqlite_database(image_features_df, identity_df)



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG if DEV_MODE else logging.INFO)
    main()
