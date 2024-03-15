"""This file is a central place to store all the paths used in the project. The functions here should be used anywhere
that a path is needed, rather than hardcoding the path somewhere else.
"""

from pathlib import Path
from loguru import logger


PROJECT_ROOT = Path(__file__).parent.parent

# INPUT DIR should contain .tif and .csv files from the camera data folder on google drive
# https://drive.google.com/drive/folders/1rU8VOIdwBuDX_N6MTn0Bg5SYYb-Ov8zv
carnegie_folder = "/carnegie/data/Shared/Labs/burlacot/Fluctuation Screen TIFF and XPIM"
if Path(carnegie_folder).exists():
    INPUT_DIR = Path(carnegie_folder)
else:
    INPUT_DIR = PROJECT_ROOT / "data"

# WELL_SEGMENTATION_DIR is where we save the output of the well segmentation as .npy files
WELL_SEGMENTATION_DIR = PROJECT_ROOT / "output" / "well_segmentation_cache"

# IDENTITY_SPREADSHEET_PATH is the path to the spreadsheet containing the plate identity information
# https://docs.google.com/spreadsheets/d/1_UcLC4jbI04Rnpt2vUkSCObX8oUY6mzl/edit?usp=drive_link&ouid=108504591016316429773&rtpof=true&sd=true
IDENTITY_SPREADSHEET_PATH = (
    PROJECT_ROOT / "data" / "plate_identity" / "burlacot_lab_plate_gene_identities.csv"
)


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
    valid_files = []
    for i, (f1, f2) in enumerate(zip(filenames_npy, filenames_meta)):
        assert f1.stem == f2.stem, f"{f1.stem} != {f2.stem}"
        try:
            assert f2.exists(), f"{f2} does not exist"
            valid_files.append(True)
        except AssertionError as e:
            logger.warning(e)
            logger.warning("Trying again with ' ' and '_' replaced")
            if " " in f2.name and "_" not in f2.name:
                f2 = f2.with_name(f2.name.replace(" ", "_"))
                try:
                    assert f2.exists(), f"{f2} still does not exist"
                    filenames_meta[i] = f2
                    valid_files.append(True)
                except AssertionError as e:
                    logger.warning(e)
                    valid_files.append(False)
                    continue
            elif "_" in f2.name and " " not in f2.name:
                f2 = f2.with_name(f2.name.replace("_", " "))
                try:
                    assert f2.exists(), f"{f2} still does not exist"
                    filenames_meta[i] = f2
                    valid_files.append(True)
                except AssertionError as e:
                    logger.warning(e)
                    valid_files.append(False)
                    continue
            elif " " in f2.name and "_" in f2.name:
                f2 = f2.with_name(f2.name.replace(" ", "_"))
                try:
                    assert f2.exists(), f"{f2} still does not exist"
                    filenames_meta[i] = f2
                    valid_files.append(True)
                except AssertionError:
                    f2 = f2.with_name(f2.name.replace("_", " "))
                    try:
                        assert f2.exists(), f"Can't find {f2} or variations of it."
                        filenames_meta[i] = f2
                        valid_files.append(True)
                    except AssertionError as e:
                        logger.warning(e)
                        valid_files.append(False)
                        continue
            else:
                logger.warning(e)
                valid_files.append(False)

    filenames_npy = [f for f, v in zip(filenames_npy, valid_files) if v]
    filenames_meta = [f for f, v in zip(filenames_meta, valid_files) if v]
    logger.info(
        f"Found {len(filenames_npy)} valid file pairs in {INPUT_DIR} and {WELL_SEGMENTATION_DIR}"
    )

    return filenames_meta, filenames_npy


def validate_inputs():
    assert INPUT_DIR.exists()
    assert len(list(INPUT_DIR.glob("*.tif"))) > 0
    assert len(list(INPUT_DIR.glob("*.tif"))) == len(set(INPUT_DIR.glob("*.tif")))
