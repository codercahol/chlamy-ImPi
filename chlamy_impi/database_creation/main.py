"""In this file, we take preprocessed image data (segmented into wells) and write out a .parquet file containing the database

The script is controlled using hard-coded constants at the top of the file. These are:
    - DEV_MODE: whether to run in development mode (only use a few files)

The database construction depends on the prior download of all .csv files into the data directory, as well as
running the well segmentation preprocessing script to generate the .npy files.

TODO: we need to also generate a text file with one line per npy/tif file, indicating processing status of this file.
"""

from itertools import product
import logging

import pandas as pd
import numpy as np

from chlamy_impi.database_creation.constants import get_possible_frame_numbers
from chlamy_impi.database_creation.construct_identity_df import construct_identity_dataframe
from chlamy_impi.database_creation.database_sanity_checks import (
    sanity_check_merged_plate_info_and_well_info,
    check_unique_plate_well_startdate,
    check_total_number_of_entries_per_plate,
    check_num_frames,
    check_all_plates_have_WT,
    check_non_null_num_mutations,
)
from chlamy_impi.database_creation.manual_error_correction import (
    remove_repeated_initial_frame,
    manually_fix_erroneous_time_points,
)
from chlamy_impi.error_correction.automated_spurious_frame_fix import fix_spurious_frames
from chlamy_impi.database_creation.utils import (
    index_to_location_rowwise,
    parse_name,
    compute_measurement_times,
    save_df_to_parquet,
)
from chlamy_impi.lib.fv_fm_functions import compute_all_fv_fm_averaged
from chlamy_impi.lib.mask_functions import compute_threshold_mask
from chlamy_impi.lib.npq_functions import compute_all_ynpq_averaged
from chlamy_impi.lib.y2_functions import (
    compute_all_y2_averaged,
    #compute_all_F_averaged,
    #compute_all_Fm_averaged,
)
from chlamy_impi.paths import (
    get_npy_and_csv_filenames,
    get_identity_spreadsheet_path,
    get_database_output_dir,
)

logger = logging.getLogger(__name__)

DEV_MODE = False
IGNORE_ERRORS = True

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


def prepare_img_array_and_df(filename_meta, filename_npy):
    """Load the image array (pre segmented into wells) and the metadata dataframe for a given plate,
    and attempt to automatically fix and spurious frame errors
    """
    img_array = np.load(filename_npy)
    meta_df = pd.read_csv(filename_meta, header=0, delimiter=";").iloc[:, :-1]

    meta_df, img_array = fix_spurious_frames(meta_df, img_array, filename_npy.stem)

    return img_array, meta_df


def construct_plate_info_df() -> pd.DataFrame:
    """
    Construct a dataframe with the logistical information about each plate.

    Columns:
    - plate
    - measurement
    - start_date
    - light_regime
    - dark_threshold
    - light_threshold
    - num_frames

    Information applies to plates (not each well)

    TODO: add remaining columns to help us control for confounders in the analysis:
        - Was there an issue?
        - Temperature under camera (avg, max, min)
        - Temperature in algae house
        - # days M plate grown
        - # days S plate grown
        - Time duration of experiment corresponding to each time point
    """

    filenames_meta, filenames_npy = get_npy_and_csv_filenames(dev_mode=DEV_MODE)

    rows = []
    failed_filenames = []

    # Print all unique plate numbers
    plate_numbers = set()
    for filename_npy in filenames_npy:
        plate_num, _, _, _ = parse_name(filename_npy.name, return_date=True)
        plate_numbers.add(plate_num)
    logger.info(f"Unique plate numbers: {plate_numbers}")

    for filename_npy, filename_meta in zip(filenames_npy, filenames_meta):
        plate_num, measurement_num, light_regime, start_date = parse_name(filename_npy.name, return_date=True)

        logger.info(f"\n\n\nProcessing plate info from filename: {filename_npy.name}")

        try:
            img_array, _ = prepare_img_array_and_df(filename_meta, filename_npy)
        except AssertionError as e:
            if IGNORE_ERRORS:
                logger.error(e)
                logger.error(f"Error processing file {filename_npy.name}. Skipping.")
                failed_filenames.append((filename_npy.name, str(e)))
                continue
            else:
                raise

        assert img_array.shape[2] % 2 == 0

        _, thresholds = compute_threshold_mask(img_array, return_thresholds=True)
        dark_threshold, light_threshold = thresholds

        row_data = {
            "plate": plate_num,
            "measurement": measurement_num,
            "start_date": start_date,
            "light_regime": light_regime,
            "dark_threshold": dark_threshold,
            "light_threshold": light_threshold,
            "num_frames": img_array.shape[2],
        }

        rows.append(row_data)

    df = pd.DataFrame(rows)

    logger.info(
        f"Constructed plate info dataframe. Shape: {df.shape}. Columns: {df.columns}."
    )
    logger.debug(f"{df.head()}")

    if failed_filenames:
        logging.error(f"Failed to process total of {len(failed_filenames)} files")
        logger.error(f"Failed to process the following files in plate info df: {failed_filenames}")

    return df


def construct_well_info_df() -> pd.DataFrame:
    """Construct a dataframe containing all the time-series data from experiments and image segmentation
     This includes image features, such as Fv/Fm, Y2, NPQ, and the times at which they were measured

     Columns:
        - plate
        - measurement
        - start_date
        - i
        - j
        - fv_fm
        - mask_area
        - y2_1, y2_2, ..., y2_81
        - ynpq_1, ynpq_2, ..., ynpq_81
        - measurement_time_0, measurement_time_2, ..., measurement_time_81
    """

    filenames_meta, filenames_npy = get_npy_and_csv_filenames(dev_mode=DEV_MODE)

    rows = []
    failed_filenames = []

    for filename_npy, filename_meta in zip(filenames_npy, filenames_meta):
        plate_num, measurement_num, light_regime, start_date = parse_name(filename_npy.name, return_date=True)

        logger.info(
            f"\n\n\nProcessing image features from filename: {filename_npy.name}"
        )

        try:
            img_array, meta_df = prepare_img_array_and_df(filename_meta, filename_npy)
        except Exception as e:
            logger.error(f"Error processing file {filename_npy.name}. Skipping.")
            logger.error(e)
            failed_filenames.append((filename_npy.name, str(e)))
            continue

        measurement_times = compute_measurement_times(meta_df=meta_df)

        assert img_array.shape[2] % 2 == 0
        assert img_array.shape[2] // 2 == len(measurement_times)

        try:
            mask_array = compute_threshold_mask(img_array)
            y2_array = compute_all_y2_averaged(img_array, mask_array)
            fv_fm_array = compute_all_fv_fm_averaged(img_array, mask_array)
            ynpq_array = compute_all_ynpq_averaged(img_array, mask_array)

            # TODO: Re-implement these functions
            # includes F0
            #F_array = compute_all_F_averaged(img_array, mask_array)
            # includes Fm at t=0
            #Fm_array = compute_all_Fm_averaged(img_array, mask_array)
        except AssertionError as e:
            if IGNORE_ERRORS:
                logger.error(f"Error computing image features for file {filename_npy.name}. Skipping.")
                failed_filenames.append((filename_npy.name, str(e)))
                continue
            else:
                raise

        Ni, Nj = img_array.shape[:2]

        for i, j in product(range(Ni), range(Nj)):
            row_data = {
                "plate": plate_num,
                "measurement": measurement_num,
                "start_date": start_date,
                "i": i,
                "j": j,
                "fv_fm": fv_fm_array[i, j],
                "mask_area": np.sum(mask_array[i, j]),
            }

            tmax = max(get_possible_frame_numbers()) - 2

            assert len(y2_array[i, j]) <= tmax - 1
            assert len(ynpq_array[i, j]) <= tmax - 1
            assert len(y2_array[i, j]) == len(ynpq_array[i, j])

            for tstep in range(1, tmax):  # There can be at most 81 time steps (82 pairs, but the first is fv/fm)
                try:
                    row_data[f"y2_{tstep}"] = y2_array[i, j, tstep - 1]
                except IndexError:
                    row_data[f"y2_{tstep}"] = np.nan

            for tstep in range(1, tmax):
                try:
                    row_data[f"ynpq_{tstep}"] = ynpq_array[i, j, tstep - 1]
                except IndexError:
                    row_data[f"ynpq_{tstep}"] = np.nan

            for k in range(tmax):
                try:
                    row_data[f"measurement_time_{k}"] = measurement_times[k]
                except IndexError:
                    row_data[f"measurement_time_{k}"] = np.nan

            # TODO: uncomment after re-implementing these lines
            #for k in range(tmax):
            #    try:
            #        row_data[f"F_{k}"] = F_array[i, j, k]
            #        row_data[f"Fm_{k}"] = Fm_array[i, j, k]
            #    except IndexError:
            #        row_data[f"F_{k}"] = np.nan
            #        row_data[f"Fm_{k}"] = np.nan

            rows.append(row_data)

    df = pd.DataFrame(rows)

    logger.info(
        f"Constructed image features dataframe. Shape: {df.shape}. Columns: {df.columns}."
    )
    logger.debug(f"{df.head()}")

    if failed_filenames:
        logger.info(f"Failed to process total of {len(failed_filenames)} files")
        logger.error(f"Failed to process the following files in well_info_df: {failed_filenames}")

    return df


def merge_plate_and_well_info_dfs(plate_df: pd.DataFrame, well_df: pd.DataFrame):
    """Merge the plate and well info dataframes. This assumes that the plate, measurement, and start_date
    columns are sufficient to uniquely identify one dataset.
    """
    df = pd.merge(
        well_df, plate_df, on=["plate", "measurement", "start_date"], how="left"
    )

    sanity_check_merged_plate_info_and_well_info(df, ignore_errors=IGNORE_ERRORS)

    # Drop rows where i or j is nan or inf
    # TODO: unsure why this has become necessary but don't have time to investigate. Worrying signal of potential bug.
    df["i"] = df["i"].replace([np.inf, -np.inf], np.nan)
    df["j"] = df["j"].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["i", "j"])
    df["i"] = df["i"].astype(int)
    df["j"] = df["j"].astype(int)

    df["well_id"] = df.apply(index_to_location_rowwise, axis=1)

    return df


def construct_gene_description_dataframe() -> pd.DataFrame:
    """extract all gene descriptions, and store as a separate dataframe

    Each gene has one description, but the descriptions are very long, so we store them separately
    """
    id_spreadsheet_path = get_identity_spreadsheet_path()
    assert id_spreadsheet_path.exists()
    df = pd.read_excel(id_spreadsheet_path, header=0, engine="openpyxl")

    # Create new dataframe with just the gene descriptions, one for each gene
    df_gene_descriptions = df[["gene", "description", "feature"]]
    df_gene_descriptions = df_gene_descriptions.drop_duplicates(subset=["gene"])

    logger.info(
        f"Constructed description dataframe. Shape: {df_gene_descriptions.shape}."
    )
    logger.debug(f"{df_gene_descriptions.head()}")
    return df_gene_descriptions


def construct_mutations_dataframe() -> pd.DataFrame:
    """Extract relevant mutant features from the identity spreadsheet

    We have columns of:
        'mutant_ID' e.g. 'LMJ.RY0401.001' or 'WT'
        'gene' e.g. 'Cre06.g278750.t1.2' or 'WT'
        'confidence_level' e.g. 5 or 10 or nan
    """
    identity_spreadsheet = get_identity_spreadsheet_path()
    df = pd.read_excel(identity_spreadsheet, header=0, engine="openpyxl")

    # TODO: verify that we don't want to include which gene feature
    # rn if we include gene features, we double count mutations to a single gene
    # if the primers picked up different regions of the gene
    df = df[["mutant_ID", "gene", "confidence_level"]]
    df = df.drop_duplicates(ignore_index=True)

    logger.info(
        f"Constructed mutation dataframe. Shape: {df.shape}. Columns: {df.columns}."
    )
    logger.debug(f"{df.head()}")
    return df


def count_wt_wells(df: pd.DataFrame) -> int:
    """Count the number of wild type wells in the dataframe"""
    return df[df.mutant_ID == "WT"].shape[0]


def get_wt_well_positions() -> list[str]:
    return ["C12", "N03", "N22"]


def write_dataframe(df: pd.DataFrame, name: str):
    """In this function, we write the dataframe to a csv file"""
    output_dir = get_database_output_dir()
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    df.to_csv(output_dir / name)

    logger.info(f"Written dataframe to {output_dir / name}")


def merge_identity_and_experimental_dfs(exptl_data, identity_df):
    # TODO - "A1" blanks often just generate NaNs
    # how to make them more useful? (we'd like them as a control)

    # First we remove plates which are not common to both dfs
    exptl_plates = exptl_data.plate.unique()
    identity_plates = identity_df.plate.unique()
    common_plates = set(exptl_plates).intersection(identity_plates)

    # Log which plates are being removed from each df
    removed_exptl_plates = set(exptl_plates) - common_plates
    removed_identity_plates = set(identity_plates) - common_plates
    logger.error(f"Removed plates from exptl_data which were not found in identity df: {removed_exptl_plates}")
    logger.error(f"Removed plates from identity_df which were not found in exptl df: {removed_identity_plates}")

    exptl_data = exptl_data[exptl_data.plate.isin(common_plates)]
    identity_df = identity_df[identity_df.plate.isin(common_plates)]

    # Verify that all ids in exptl data are present in identity df except *-A1
    non_blank_wells = exptl_data.well_id[exptl_data.well_id != "A01"]
    exptl_plate_n_well = set(product(exptl_data.plate, non_blank_wells))
    identity_plate_n_well = set(product(identity_df.plate, identity_df.well_id))

    if IGNORE_ERRORS:
        # Ignore all wells which are not in the identity dataframe
        exptl_plate_n_well = exptl_plate_n_well.intersection(identity_plate_n_well)
        exptl_data = exptl_data[exptl_data[['plate', 'well_id']].apply(tuple, axis=1).isin(exptl_plate_n_well)]
        logger.error(f"Removed {len(exptl_plate_n_well) - len(exptl_data)} wells from exptl_data which were not present in identity df")
    else:
        # Raise error if a well is not in the identity dataframe
        err_msg = exptl_plate_n_well - identity_plate_n_well
        assert exptl_plate_n_well.issubset(identity_plate_n_well), err_msg

    total_df = pd.merge(
        exptl_data, identity_df, on=["plate", "well_id"], how="left", validate="m:1"
    )
    logger.info(f"Shape of total_df: {total_df.shape}, Columns: {total_df.columns}")
    logger.debug(total_df.head())

    logger.info(f"After merge, we have data for plates: {total_df.plate.unique()}")
    logger.info(
        f"After merge, we have data for light regimes: {total_df.light_regime.unique()}"
    )
    logger.info(
        f"After merge, we have data for measurement numbers: {total_df.measurement.unique()}"
    )

    return total_df


def final_df_sanity_checks(df: pd.DataFrame):
    """Final set of tests applied to the dataframe before we write it to parquet file"""
    check_unique_plate_well_startdate(df)
    check_total_number_of_entries_per_plate(df)
    check_num_frames(df)
    check_all_plates_have_WT(df)
    check_non_null_num_mutations(df)


def print_final_df_stats(df: pd.DataFrame):
    """Print out some information about this dataframe
    """
    unique_combinations = df.groupby(["plate", "measurement", "start_date"]).size()
    logger.info(f"Number of unique plate / measurement / date combinations: {len(unique_combinations)}")


def main():
    mutations_df = construct_mutations_dataframe()
    logger.info(f"Constructed mutations dataframe. Shape: {mutations_df.shape}.")
    identity_df = construct_identity_dataframe(mutations_df)
    logger.info(f"Constructed identity dataframe. Shape: {identity_df.shape}.")

    logger.info("Constructing gene descriptions dataframe...")
    gene_descriptions = construct_gene_description_dataframe()
    write_dataframe(gene_descriptions, f"gene_descriptions.csv")

    plate_data = construct_plate_info_df()
    logger.info(f"Constructed plate info dataframe. Shape: {plate_data.shape}.")
    well_data = construct_well_info_df()
    logger.info(f"Constructed well info dataframe. Shape: {well_data.shape}.")
    exptl_data = merge_plate_and_well_info_dfs(well_data, plate_data)
    logger.info(f"Constructed merged dataframe. Shape: {exptl_data.shape}.")

    total_df = merge_identity_and_experimental_dfs(exptl_data, identity_df)
    logger.info(f"Constructed total dataframe. Shape: {total_df.shape}.")

    logger.info("Sanity checks on final dataframe...")
    final_df_sanity_checks(total_df)
    logger.info("All sanity checks passed.")
    print_final_df_stats(total_df)

    logger.info("Writing dataframe to parquet file...")
    save_df_to_parquet(total_df)
    logger.info("Successfully generated new database files!!!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG if DEV_MODE else logging.INFO)
    main()
