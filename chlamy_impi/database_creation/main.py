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
# %%
%load_ext autoreload
%autoreload 2

import datetime
from itertools import product
from pathlib import Path
import logging
from typing import List, Optional

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from chlamy_impi.database_creation.error_correction import (
    fix_erroneous_time_points,
    remove_repeated_initial_frame,
)
from chlamy_impi.database_creation.utils import (
    location_to_index,
    index_to_location_rowwise,
    parse_name,
    spreadsheet_plate_to_numeric,
    compute_measurement_times,
)
from chlamy_impi.lib.fv_fm_functions import compute_all_fv_fm_averaged
from chlamy_impi.lib.mask_functions import compute_threshold_mask
from chlamy_impi.lib.npq_functions import compute_all_npq_averaged
from chlamy_impi.lib.y2_functions import compute_all_y2_averaged

logger = logging.getLogger(__name__)

DEV_MODE = False

INPUT_DIR = Path("../../data")
IDENTITY_SPREADSHEET_PATH = Path(
    "../../data/Identity plates in Burlacot Lab 20231221 simplified.xlsx - large-lib_rearray2.txt.csv"
)
OUTPUT_DIR = Path("./../../output/database_creation/v2")


def prepare_img_array_and_df(filename_meta, filename_npy):
    img_array = np.load(filename_npy)
    img_array = remove_repeated_initial_frame(img_array)
    meta_df = pd.read_csv(filename_meta, header=0, delimiter=";").iloc[:, :-1]
    meta_df, img_array = fix_erroneous_time_points(meta_df, img_array)
    return img_array, meta_df


def get_filenames(input_dir: Path):
    assert input_dir.exists()

    filenames_npy = list(input_dir.glob("*.npy"))
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


def construct_exptl_data_df(input_dir: Path) -> pd.DataFrame:
    """In this function, we construct a dataframe containing all the
     experimental data from experiments and image segmentation
      This includes image features, such as Fv/Fm, Y2, NPQ

    This also includes:
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
     TODO:
     - Other quantifiers of fluorescence or shape heterogeneity
    """

    filenames_meta, filenames_npy = get_filenames(input_dir)

    rows = []

    for filename_npy, filename_meta in zip(filenames_npy, filenames_meta):
        plate_num, measurement_num, light_regime = parse_name(filename_npy)

        logger.info(
            f"\n\n\nProcessing image features from filename: {filename_npy.name}"
        )

        img_array, meta_df = prepare_img_array_and_df(filename_meta, filename_npy)

        measurement_times = compute_measurement_times(meta_df=meta_df)

        assert img_array.shape[2] % 2 == 0

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
                "mask_area": np.sum(mask_array[i, j]),
                "light_regime": light_regime,
                "threshold": threshold,
                "num_frames": img_array.shape[2],
            }

            assert len(y2_array[i, j]) <= 81
            assert len(npq_array[i, j]) <= 81
            assert len(y2_array[i, j]) == len(npq_array[i, j])

            for tstep in range(
                1, 82
            ):  # There can be at most 81 time steps (82 pairs, but the first is fv/fm)
                try:
                    row_data[f"y2_{tstep}"] = y2_array[i, j, tstep - 1]
                except IndexError:
                    row_data[f"y2_{tstep}"] = np.nan

            for tstep in range(1, 82):
                try:
                    row_data[f"npq_{tstep}"] = npq_array[i, j, tstep - 1]
                except IndexError:
                    row_data[f"npq_{tstep}"] = np.nan

            for k in range(82):
                try:
                    row_data[f"measurement_time_{k}"] = measurement_times[k]
                except IndexError:
                    row_data[f"measurement_time_{k}"] = np.nan

            rows.append(row_data)

    df = pd.DataFrame(rows)
    df["id"] = df.apply(
        lambda x: "{}-{}".format(x.plate, index_to_location_rowwise(x)), axis=1
    )
    df = df.set_index("id")

    logger.info(
        f"Constructed image features dataframe. Shape: {df.shape}. Columns: {df.columns}."
    )
    logger.info(f"{df.head()}")
    return df


def construct_gene_description_dataframe(identity_spreadsheet: Path) -> pd.DataFrame:
    """In this function, we extract all gene descriptions, and store as a separate dataframe

    Each gene has one description, but the descriptions are very long, so we store them separately
    """
    assert IDENTITY_SPREADSHEET_PATH.exists()
    df = pd.read_csv(IDENTITY_SPREADSHEET_PATH, header=0)

    # Create new dataframe with just the gene descriptions, one for each gene
    df_gene_descriptions = df[["gene", "description"]]
    df_gene_descriptions = df_gene_descriptions.drop_duplicates(subset=["gene"])

    logger.info(
        f"Constructed description dataframe. Shape: {df_gene_descriptions.shape}."
    )
    logger.info(f"{df_gene_descriptions.head()}")
    return df_gene_descriptions


def construct_mutations_dataframe(identity_spreadsheet: Path) -> pd.DataFrame:
    """In this function, we extract all mutation features, such as gene name, location, etc.

    This is a separate table because each mutant_ID can have several mutations
    """
    df = pd.read_csv(identity_spreadsheet, header=0)

    # TODO: verify that we don't want to include which gene feature
    # rn if we include gene features, we double count mutations to a single gene
    # if the primers picked up different regions of the gene
    df = df[["mutant_ID", "gene", "confidence_level"]]
    df = df.drop_duplicates(ignore_index=True)

    logger.info(
        f"Constructed mutation dataframe. Shape: {df.shape}. Columns: {df.columns}."
    )
    logger.info(f"{df.head()}")
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

    # set a unique index
    df["id"] = df.apply(lambda x: "{}-{}".format(x.plate, x.location), axis=1)
    df = df.drop(columns=["location", "plate"])
    df = df.drop_duplicates(ignore_index=True)
    
    # Add column which tells us the number of genes which were mutated
    signif_mutations = mutation_df[mutation_df.confidence_level <= conf_threshold]
    gene_mutation_counts = signif_mutations.groupby("mutant_ID").nunique()["gene"]

    mutated_genes = signif_mutations.groupby("mutant_ID").apply(
        lambda x: ",".join(set(x.gene))
    )
    mutated_genes = mutated_genes.reset_index().rename(columns={0: "mutated_genes"})
    df = pd.merge(df, mutated_genes, on="mutant_ID")
    df = df.set_index("id")

    df["num_mutations"] = df["mutant_ID"].apply(lambda x: gene_mutation_counts[x])

    logger.info(
        f"Constructed identity dataframe. Shape: {df.shape}. Columns: {df.columns}."
    )
    return df


def write_dataframe(df: pd.DataFrame, name: str, output_dir: Path = OUTPUT_DIR):
    """In this function, we write the dataframe to a csv file"""
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    df.to_csv(output_dir / name)

    logger.info(f"Written dataframe to {OUTPUT_DIR / name}")


def save_df_to_parquet(df: pd.DataFrame, filename: str, output_dir: Path = OUTPUT_DIR):
    table = pa.Table.from_pandas(df)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    filename = filename + ".parquet"
    pq.write_table(table, output_dir / filename)
    logger.info("File saved at: {}".format(output_dir / filename))

def read_df_from_parquet(
        filename: str, columns: Optional[List[str]] = None
) -> pd.DataFrame:
    table = pq.read_table(filename, columns = columns)
    df = table.to_pandas()
    return df

def main():
    exptl_data = construct_exptl_data_df(INPUT_DIR)
    mutations_df = construct_mutations_dataframe(IDENTITY_SPREADSHEET_PATH)
    identity_df = construct_identity_dataframe(IDENTITY_SPREADSHEET_PATH, mutations_df)

    total_df = pd.merge(exptl_data, identity_df, on="id")
    save_df_to_parquet(total_df, "db")
    df = read_df_from_parquet(OUTPUT_DIR / "db.parquet")

    gene_descriptions = construct_gene_description_dataframe(IDENTITY_SPREADSHEET_PATH)
    write_dataframe(gene_descriptions, f"gene_descriptions.csv")


# %%

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG if DEV_MODE else logging.INFO)
    main()
