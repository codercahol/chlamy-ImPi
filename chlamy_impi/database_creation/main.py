"""In this file, we take preprocessed image data (segmented into wells) and construct a database (currently csv files and sqlite)
containing all the features we need for our analysis.

The script is controlled using hard-coded constants at the top of the file. These are:
    - DEV_MODE: whether to run in development mode (only use a few files)
    - INPUT_DIR: directory containing the preprocessed image data
    - IDENTITY_SPREADSHEET_PATH: path to the spreadsheet containing information about the identity of each well.
        This is found at: https://docs.google.com/spreadsheets/d/1_UcLC4jbI04Rnpt2vUkSCObX8oUY6mzl/edit#gid=206647583
    - OUTPUT_DIR: directory to write the csv files to

The database will have four tables:
    - image_features: contains features extracted from the images, such as Fv/Fm, Y2, NPQ, along with experimental information
    - identity: contains information about the identity of each mutant, such as well location, plate number, etc.
    - mutations: contains information about the mutations in each mutant, such as disrupted gene name, type, confidence level, etc.
    - gene_descriptions: contains lengthy descriptions of each gene
"""

from itertools import product
from pathlib import Path
import logging
import sqlite3

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from chlamy_impi.database_creation.utils import (
    location_to_index,
    parse_name,
    spreadsheet_plate_to_numeric,
)
from chlamy_impi.lib.fv_fm_functions import compute_all_fv_fm_averaged
from chlamy_impi.lib.mask_functions import compute_threshold_mask
from chlamy_impi.lib.npq_functions import compute_all_npq_averaged
from chlamy_impi.lib.y2_functions import compute_all_y2_averaged

logger = logging.getLogger(__name__)

DEV_MODE = False
INPUT_DIR = Path("../../output/image_processing/v6/img_array")
IDENTITY_SPREADSHEET_PATH = Path(
    "../../data/Identity plates in Burlacot Lab 20231221 simplified.xlsx - large-lib_rearray2.txt.csv"
)
OUTPUT_DIR = Path("./../../output/database_creation/v1")


def construct_img_feature_dataframe(input_dir: Path):
    """In this function, we extract all image features, such as Fv/Fm, Y2, NPQ

    TODO: add remaining experimental columns:
     - Start time and date
     - Was there an issue?
     - Temperature under camera (avg, max, min)
     - Temperature in algae house
     - # days M plate grown
     - # days S plate grown
     - Other quantifiers of fluorescence or shape heterogeneity
    """
    assert input_dir.exists()
    filenames = list(input_dir.glob("*.npy"))

    if DEV_MODE:
        filenames = filenames[:1]
        logger.info(f"DEV_MODE: only using {len(filenames)} files")

    rows = []

    for filename in filenames:
        plate_num, measurement_num = parse_name(filename)

        logger.info(f"Processing image features from filename: {filename.name}")

        img_array = np.load(filename)

        if img_array.shape[2] % 2 != 0:
            logger.warning(
                f"Odd number of time steps ({img_array.shape[2]}), removing last time step"
            )
            img_array = img_array[
                :, :, :-1, ...
            ]  # We rely on an even number of time steps, to pair up dark and light images for NPQ and Y2

        mask_array, threshold = compute_threshold_mask(img_array, return_threshold=True)
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
                "threshold": threshold,
                "mask_area": np.sum(mask_array[i, j]),
            }

            assert len(y2_array[i, j]) <= 83
            assert len(npq_array[i, j]) <= 83
            assert len(y2_array[i, j]) == len(npq_array[i, j])

            for tstep in range(83):  # There can be at most 168/2 - 1 = 83 time steps
                try:
                    row_data[f"y2_{tstep}"] = y2_array[i, j, tstep]
                    row_data[f"npq_{tstep}"] = npq_array[i, j, tstep]
                except IndexError:
                    row_data[f"y2_{tstep}"] = np.nan
                    row_data[f"npq_{tstep}"] = np.nan

            rows.append(row_data)

    df = pd.DataFrame(rows)
    df = df.set_index(["plate", "i", "j"])

    logger.info(
        f"Constructed image features dataframe. Shape: {df.shape}. Columns: {df.columns}."
    )
    return df


def construct_gene_description_dataframe(identity_spreadsheet: Path) -> pd.DataFrame:
    """In this function, we extract all gene descriptions, and store as a separate dataframe

    Each gene has one description, but the descriptions are very long, so we store them separately
    """
    df = pd.read_csv(identity_spreadsheet, header=0)

    # Create new dataframe with just the gene descriptions, one for each gene
    df_gene_descriptions = df[["gene", "description"]]
    df_gene_descriptions = df_gene_descriptions.drop_duplicates(subset=["gene"])

    logger.info(
        f"Constructed description dataframe. Shape: {df_gene_descriptions.shape}."
    )
    return df_gene_descriptions


def construct_mutations_dataframe() -> pd.DataFrame:
    """In this function, we extract all mutation features, such as gene name, location, etc.

    This is a separate table because each mutant_ID can have several mutations
    """
    df = pd.read_csv(IDENTITY_SPREADSHEET_PATH, header=0)

    df = df[["mutant_ID", "gene", "feature", "confidence_level"]]
    df = df.drop_duplicates(ignore_index=True)

    logger.info(
        f"Constructed mutation dataframe. Shape: {df.shape}. Columns: {df.columns}."
    )
    return df


def construct_identity_dataframe(
    identity_spreadsheet: Path, mutation_df: pd.DataFrame, conf_threshold: int = 5
) -> pd.DataFrame:
    """In this function, we extract all identity features, such as strain name, location, etc.

    There is a single row per plate-well ID
    (currently this corresponds also to a unique mutant ID, but won't always)
    """
    df = pd.read_csv(identity_spreadsheet, header=0)

    # Assert that there are no null values in the "Location" and "New Location" columns
    assert df["Location"].notnull().all()
    assert df["New location"].notnull().all()

    # Collect columns which we need
    df = df.rename(columns={"New location": "plate", "Location": "location"})
    df = df[["mutant_ID", "plate", "location"]]

    df["plate"] = df["plate"].apply(spreadsheet_plate_to_numeric)
    df["row_idx"] = df["location"].apply(lambda x: location_to_index(x)[0])
    df["col_idx"] = df["location"].apply(lambda x: location_to_index(x)[1])

    # set a unique index
    df["id"] = df.apply(lambda x: "{}-{}}".format(x.plate, x.location), axis=1)
    df = df.drop(columns=["location"])
    df = df.drop_duplicates(ignore_index=True)
    df.set_index("id")

    # Add column which tells us the number of genes which were mutated
    gene_mutation_counts = mutation_df.groupby("mutant_ID").count()["gene"]

    signif_mutations = mutation_df[mutation_df.confidence_level <= conf_threshold]
    mutated_genes = signif_mutations.groupby("mutant_ID").apply(
        lambda x: ",".join(set(x.gene))
    )
    mutated_genes = mutated_genes.reset_index().rename(columns={0: "mutated_genes"})
    df = pd.merge(df, mutated_genes, on="mutant_ID")

    df["num_mutations"] = df["mutant_ID"].apply(lambda x: gene_mutation_counts[x])

    logger.info(
        f"Constructed identity dataframe. Shape: {df.shape}. Columns: {df.columns}."
    )
    return df


# goal: do prelim anal of T.S. using parquet
# goal: organize the saved parquet files more to get the other datatypes needed


def write_dataframe(df: pd.DataFrame, name: str):
    """In this function, we write the dataframe to a csv file"""
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)
    df.to_csv(OUTPUT_DIR / name)

    logger.info(f"Written dataframe to {OUTPUT_DIR / name}")


def create_sqlite_database(name_to_df: dict[str, pd.DataFrame]):
    """In this function, we create a sqlite database from the two dataframes"""
    conn = sqlite3.connect(OUTPUT_DIR / "chlamy_screen_database.db")
    for name, df in name_to_df.items():
        df.to_sql(name, conn, if_exists="replace")
    conn.close()


def save_df_to_parquet():
    return


def main():
    image_features_df = construct_img_feature_dataframe(INPUT_DIR)
    mutations_df = construct_mutations_dataframe()
    identity_df = construct_identity_dataframe(mutations_df)

    gene_descriptions = construct_gene_description_dataframe(IDENTITY_SPREADSHEET_PATH)

    name_to_df = {
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
