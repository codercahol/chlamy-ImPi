"""This file is a central place to store all the paths used in the project. The functions here should be used anywhere
that a path is needed, rather than hardcoding the path somewhere else.
"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).parent.parent

# INPUT DIR should contain .tif and .csv files from the camera data folder on google drive
# https://drive.google.com/drive/folders/1rU8VOIdwBuDX_N6MTn0Bg5SYYb-Ov8zv
INPUT_DIR = PROJECT_ROOT / "data_13062024"

# WELL_SEGMENTATION_DIR is where we save the output of the well segmentation as .npy files
WELL_SEGMENTATION_DIR = PROJECT_ROOT / "output" / "well_segmentation_cache"

# IDENTITY_SPREADSHEET_PATH is the path to the spreadsheet containing the plate identity information
# https://docs.google.com/spreadsheets/d/1_UcLC4jbI04Rnpt2vUkSCObX8oUY6mzl/edit?usp=drive_link&ouid=108504591016316429773&rtpof=true&sd=true
# Update - final sheet is now here:
# https://docs.google.com/spreadsheets/d/1reX1t-C9rwjwhJWRowGZPUV7F4B1Wvvw/edit#gid=1935584839
IDENTITY_SPREADSHEET_PATH = \
    INPUT_DIR / "Finalized Identities Phase I plates.xlsx"


# DATABASE_DIR is where we save the output of the database creation as .csv and parquet files
DATABASE_DIR = PROJECT_ROOT / "output" / "database_creation"


def find_all_tif_images():
    return list(INPUT_DIR.glob("*.tif"))


def well_segmentation_output_dir_path(name) -> Path:
    savedir = WELL_SEGMENTATION_DIR / name
    return savedir


def well_segmentation_visualisation_dir_path(name) -> Path:
    savedir = well_segmentation_output_dir_path(name) / "visualisation_raw"
    return savedir


def well_segmentation_histogram_dir_path(name) -> Path:
    savedir = well_segmentation_output_dir_path(name) / "visualisation_histograms"
    return savedir


def npy_img_array_path(name):
    return WELL_SEGMENTATION_DIR / f"{name}.npy"


def get_identity_spreadsheet_path():
    return IDENTITY_SPREADSHEET_PATH


def get_database_output_dir():
    return DATABASE_DIR


def get_parquet_filename():
    return DATABASE_DIR / "database.parquet"


def get_npy_and_csv_filenames(dev_mode: bool = False):
    """In this function, we get a list of all the .npy and .csv files in the input directory

    We also check that the two lists of filenames are the same, and that the .csv files exist
    """
    assert WELL_SEGMENTATION_DIR.exists()
    assert INPUT_DIR.exists()

    filenames_npy = list(WELL_SEGMENTATION_DIR.glob("*.npy"))
    filenames_npy.sort()

    filenames_meta = [INPUT_DIR / x.with_suffix(".csv").name for x in filenames_npy]

    if dev_mode:
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


def get_npy_and_csv_filenames_given_basename(basename: str) -> tuple[pd.DataFrame, np.array]:
    """Given a base name, load the corresponding meta df and image array
    """
    meta_df = pd.read_csv(INPUT_DIR / f"{basename}.csv", header=0, delimiter=";").iloc[:, :-1]
    img_array = np.load(WELL_SEGMENTATION_DIR / f"{basename}.npy")

    return meta_df, img_array


def validate_inputs():
    assert INPUT_DIR.exists()
    assert len(list(INPUT_DIR.glob("*.tif"))) > 0
    assert len(list(INPUT_DIR.glob("*.tif"))) == len(set(INPUT_DIR.glob("*.tif")))
