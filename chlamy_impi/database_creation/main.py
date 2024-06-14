"""In this file, we take preprocessed image data (segmented into wells) and write out a .parquet file containing the database

The script is controlled using hard-coded constants at the top of the file. These are:
    - DEV_MODE: whether to run in development mode (only use a few files)

The database construction depends on the prior download of all .csv files into the data directory, as well as
running the well segmentation preprocessing script to generate the .npy files.

"""

from itertools import product
import logging

import pandas as pd
import numpy as np

from chlamy_impi.database_creation.database_sanity_checks import sanity_check_merged_plate_info_and_well_info, \
    check_num_mutations, check_unique_plate_well_startdate, \
    check_total_number_of_entries_per_plate, check_plate_and_wells_are_unique, check_num_frames, \
    check_all_plates_have_WT, check_non_null_num_mutations
from chlamy_impi.database_creation.error_correction import (
    remove_repeated_initial_frame, manually_fix_erroneous_time_points,
)
from chlamy_impi.database_creation.utils import (
    index_to_location_rowwise,
    parse_name,
    spreadsheet_plate_to_numeric,
    compute_measurement_times,
    save_df_to_parquet,
)
from chlamy_impi.lib.fv_fm_functions import compute_all_fv_fm_averaged
from chlamy_impi.lib.mask_functions import compute_threshold_mask
from chlamy_impi.lib.npq_functions import compute_all_npq_averaged
from chlamy_impi.lib.y2_functions import compute_all_y2_averaged
from chlamy_impi.paths import (
    get_npy_and_csv_filenames,
    get_identity_spreadsheet_path,
    get_database_output_dir,
)

logger = logging.getLogger(__name__)

DEV_MODE = False
IGNORE_ERRORS = True

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



def prepare_img_array_and_df(filename_meta, filename_npy):
    img_array = np.load(filename_npy)
    img_array = remove_repeated_initial_frame(img_array)
    meta_df = pd.read_csv(filename_meta, header=0, delimiter=";").iloc[:, :-1]
    meta_df, img_array = manually_fix_erroneous_time_points(meta_df, img_array, filename_npy.stem)
    return img_array, meta_df


def construct_plate_info_df() -> pd.DataFrame:
    """construct a dataframe with the
    logistical information about each plate, such as:
    - Plate number
    - Start date
    - Light regime
    - Threshold of masks
    - Number of frames

    Information applies to plates (not each well)

    TODO: add remaining columns:
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

    for filename_npy, filename_meta in zip(filenames_npy, filenames_meta):
        try:
            plate_num, measurement_num, light_regime, start_date = parse_name(filename_npy.name, return_date=True)
        except AssertionError as e:
            if IGNORE_ERRORS:
                logger.error(e)
                logger.error(f"Error parsing name of file {filename_npy.name}. Skipping.")
                failed_filenames.append((filename_npy.name, str(e)))
                continue
            else:
                raise

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
        logger.error(f"Failed to process the following files: {failed_filenames}")

    return df


def construct_well_info_df() -> pd.DataFrame:
    """construct a dataframe containing all the
    time-series data from experiments and image segmentation
     This includes image features, such as Fv/Fm, Y2, NPQ, and the times at which they were measured

    TODO:
    - Other quantifiers of fluorescence or shape heterogeneity
    """

    filenames_meta, filenames_npy = get_npy_and_csv_filenames(dev_mode=DEV_MODE)

    rows = []
    failed_filenames = []

    for filename_npy, filename_meta in zip(filenames_npy, filenames_meta):
        try:
            plate_num, measurement_num, light_regime, start_date = parse_name(filename_npy.name, return_date=True)
        except AssertionError:
            if IGNORE_ERRORS:
                logger.error(f"Error parsing name of file {filename_npy.name}. Skipping.")
                failed_filenames.append(filename_npy.name)
                continue
            else:
                raise

        logger.info(
            f"\n\n\nProcessing image features from filename: {filename_npy.name}"
        )

        try:
            img_array, meta_df = prepare_img_array_and_df(filename_meta, filename_npy)
        except Exception as e:
            logger.error(f"Error processing file {filename_npy.name}. Skipping.")
            logger.error(e)
            failed_filenames.append(filename_npy.name)
            continue

        measurement_times = compute_measurement_times(meta_df=meta_df)

        assert img_array.shape[2] % 2 == 0
        assert img_array.shape[2] // 2 == len(measurement_times)

        try:
            mask_array = compute_threshold_mask(img_array)
            y2_array = compute_all_y2_averaged(img_array, mask_array)
            fv_fm_array = compute_all_fv_fm_averaged(img_array, mask_array)
            npq_array = compute_all_npq_averaged(img_array, mask_array)
        except AssertionError:
            if IGNORE_ERRORS:
                logger.error(f"Error computing image features for file {filename_npy.name}. Skipping.")
                continue
                failed_filenames.append(filename_npy.name)
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

    logger.info(
        f"Constructed image features dataframe. Shape: {df.shape}. Columns: {df.columns}."
    )
    logger.debug(f"{df.head()}")

    logger.error(f"Failed to process the following files: {failed_filenames}")

    return df


def merge_plate_and_well_info_dfs(plate_df: pd.DataFrame, well_df: pd.DataFrame):
    """Merge the plate and well info dataframes. This assumes that the plate, measurement, and start_date
    columns are sufficient to uniquely identify one dataset.
    """
    df = pd.merge(well_df, plate_df, on=["plate", "measurement", "start_date"], how="left")

    sanity_check_merged_plate_info_and_well_info(df, ignore_errors=IGNORE_ERRORS)

    # Drop rows where i or j is nan or inf
    # TODO: unsure why this has become necessary but don't have time to investigate. Worrying signal of potential bug.
    df["i"] = df["i"].replace([np.inf, -np.inf], np.nan)
    df["j"] = df["j"].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["i", "j"])
    df["i"] = df["i"].astype(int)
    df["j"] = df["j"].astype(int)

    df["well_id"] = df.apply(index_to_location_rowwise, axis=1)

    logger.info(
        f"Constructed merged dataframe. Shape: {df.shape}."
    )
    return df


def construct_gene_description_dataframe() -> pd.DataFrame:
    """extract all gene descriptions, and store as a separate dataframe

    Each gene has one description, but the descriptions are very long, so we store them separately
    """
    id_spreadsheet_path = get_identity_spreadsheet_path()
    assert id_spreadsheet_path.exists()
    df = pd.read_excel(id_spreadsheet_path, header=0, engine='openpyxl')

    # Create new dataframe with just the gene descriptions, one for each gene
    df_gene_descriptions = df[["gene", "description"]]
    df_gene_descriptions = df_gene_descriptions.drop_duplicates(subset=["gene"])

    logger.info(
        f"Constructed description dataframe. Shape: {df_gene_descriptions.shape}."
    )
    logger.debug(f"{df_gene_descriptions.head()}")
    return df_gene_descriptions


def construct_mutations_dataframe() -> pd.DataFrame:
    """extract all mutation features, such as gene name, location, etc.

    This is a separate table because each mutant_ID can have several mutations
    """
    identity_spreadsheet = get_identity_spreadsheet_path()
    df = pd.read_excel(identity_spreadsheet, header=0, engine='openpyxl')

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


def construct_identity_dataframe(
    mutation_df: pd.DataFrame, conf_threshold: int = 5
) -> pd.DataFrame:
    """extract all identity features, such as strain name, location, etc.

    There is a single row per plate-well ID
    (currently this corresponds also to a unique mutant ID, but won't always)

    We create columns as follows:
        - id: unique identifier for each well such as 1-A1, 12-F12, etc. (index)
        - mutant_ID: unique identifier for each mutant
        - num_mutations: number of genes which were mutated
        - mutated_genes: string of comma-separated gene names which were mutated

    """
    identity_spreadsheet = get_identity_spreadsheet_path()
    df = pd.read_excel(identity_spreadsheet, header=0, engine='openpyxl')

    assert df["mutant_ID"].notnull().all(), f'Found a total of {df["mutant_ID"].isnull().sum()} null values in mutant_ID'

    # NOTE: the finalised identity spreadsheet has non-unique column names. Pandas appends .1, .2, .3, etc. to these

    # In the old spreadsheet, Location was e.g. A10, and New location was e.g. Plate 01. These have been mapped to the
    # columns below.
    # The new spreadsheet has some null values for "New Location" and "New Location.4" which we need to drop
    df = df.dropna(subset=["New Location", "New Location.4"])

    # Drop the final row which just says "Z-END"
    df = df.iloc[:-1]

    # Map all values of "Plate RTL" to "Plate 98" to keep with the numeric plate number format
    df["New Location"] = df["New Location"].apply(lambda x: x.replace("Plate RTL", "Plate 98"))

    # Map all values of "Plate 1" to "Plate 01" in the "New Location" column
    df["New Location"] = df["New Location"].apply(lambda x: x.replace("Plate 1", "Plate 01") if x == "Plate 1" else x)

    # Check that all entries in the "New Location" column are of the form "Plate XX"
    assert df["New Location"].apply(lambda x: x.startswith("Plate ")).all(), df["New Location"].unique()
    assert df["New Location"].apply(lambda x: len(x) == 8).all(), df["New Location"].unique()
    assert df["New Location"].apply(lambda x: x[6:].isdigit()).all(), df["New Location"].unique()

    # Check that all entries in the "New Location.4" column are of the form "A01", "B12", etc.
    assert df["New Location.4"].apply(lambda x: len(x) == 3).all(), df["New Location.4"].unique()
    assert df["New Location.4"].apply(lambda x: x[0] in "ABCDEFGHIJKLMNOP").all(), df["New Location.4"].unique()
    assert df["New Location.4"].apply(lambda x: x[1:].isdigit()).all(), df["New Location.4"].unique()

    # Collect columns which we need
    df = df.rename(columns={"New Location": "plate", "New Location.4": "well_id"})
    df["plate"] = df["plate"].apply(spreadsheet_plate_to_numeric)
    df_features = df[["mutant_ID", "plate", "well_id", "feature"]]
    df = df[["mutant_ID", "plate", "well_id"]]

    df = df.drop_duplicates(ignore_index=True)
    df_features = df_features.drop_duplicates(ignore_index=True)
    df_features = df_features.dropna(subset=["feature"])

    # Concatenate all features into a single string, and place into feature column
    df_grouped = df_features.groupby(["mutant_ID", "plate", "well_id"]).apply(
        lambda x: ",".join(set(x.feature)))

    # Convert df_grouped back into a dataframe - the index is a multi-index of (mutant_ID, plate, well_id)
    df_grouped = df_grouped.reset_index().rename(columns={0: "feature"})

    # Merge the cleaned features back in
    df = pd.merge(df, df_grouped, on=["mutant_ID", "plate", "well_id"], how="left")

    # Check that all mutant_IDs are unique - print out any duplicates
    check_plate_and_wells_are_unique(df)

    df = add_mutated_genes_col(conf_threshold, df, mutation_df)

    # Add rows for wild type plate 99
    wt_rows = create_wt_rows()
    df_wt = pd.DataFrame(wt_rows)
    df = pd.concat([df, df_wt], axis=0, ignore_index=False)

    check_plate_and_wells_are_unique(df)
    assert df["mutant_ID"].notnull().all(), f'Found a total of {df["mutant_ID"].isnull().sum()} null values in mutant_ID'
    check_num_mutations(df)

    # Group by the plate number (the first part of the index string) to check number of wells per plate
    plates = df.plate
    plate_counts = plates.value_counts()
    for plate, count in plate_counts.items():
        assert count <= 384, f"Plate {plate} has {count} wells with ids {df[df.plate == plate].well_id.unique()}"

    logger.info(
        f"Constructed identity dataframe. Shape: {df.shape}. Columns: {df.columns}."
    )
    logger.info(f"Values of num_mutations: {df.num_mutations.unique()}")
    logger.debug(f"{df.head()}")

    return df


def add_mutated_genes_col(conf_threshold: float, df: pd.DataFrame, mutation_df: pd.DataFrame) -> pd.DataFrame:
    """Add column which tells us the number of genes which were mutated, as well as comma separated list of genes
    """
    num_rows = len(df)

    signif_mutations = mutation_df[mutation_df.confidence_level <= conf_threshold]
    gene_mutation_counts = signif_mutations.groupby("mutant_ID").nunique()["gene"]
    mutated_genes = signif_mutations.groupby("mutant_ID").apply(
        lambda x: ",".join(set(x.gene))
    )
    mutated_genes = mutated_genes.reset_index().rename(columns={0: "mutated_genes"})
    df = pd.merge(df, mutated_genes, on="mutant_ID", how='left')
    df["num_mutations"] = df["mutant_ID"].apply(lambda x: gene_mutation_counts.get(x, 0))

    assert len(df) == num_rows, f"Length of dataframe changed from {num_rows} to {len(df)}"
    check_plate_and_wells_are_unique(df)
    assert df["mutant_ID"].notnull().all(), f'Found a total of {df["mutant_ID"].isnull().sum()} null values in mutant_ID'

    return df


def create_wt_rows() -> list[dict]:
    """In this function, we create rows for the wild type plate 99"""
    rows = []
    for well in well_position_iterator():
        row_data = {
            "plate": 99,
            "well_id": well,
            "mutant_ID": "WT",
            "num_mutations": 0,
            "mutated_genes": "",
        }
        rows.append(row_data)
    return rows


def get_wt_well_positions() -> list[str]:
    return ["C12", "N03", "N22"]


def well_position_iterator():
    for i in range(1, 17):
        for j in range(1, 25):
            yield f"{chr(ord('A') + i - 1)}{j:02d}"


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

    # Verify that all ids in exptl data are present in identity df except *-A1
    non_blank_wells = exptl_data.well_id[exptl_data.well_id != "A01"]
    exptl_plate_n_well = set(product(exptl_data.plate, non_blank_wells))
    identity_plate_n_well = set(product(identity_df.plate, identity_df.well_id))

    if IGNORE_ERRORS:
        # Ignore all wells which are not in the identity dataframe
        exptl_plate_n_well = exptl_plate_n_well.intersection(identity_plate_n_well)
        exptl_data = exptl_data[exptl_data[['plate', 'well_id']].apply(tuple, axis=1).isin(exptl_plate_n_well)]
        logger.error(f"Removed {len(exptl_plate_n_well) - len(exptl_data)} wells from exptl_data which were not present in idneity df")
    else:
        err_msg = exptl_plate_n_well - identity_plate_n_well
        assert exptl_plate_n_well.issubset(identity_plate_n_well), err_msg

    total_df = pd.merge(exptl_data, identity_df, on=["plate", "well_id"], how="left", validate="m:1")
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
    """Final set of tests applied to the dataframe before we write it to parquet file
    """
    check_unique_plate_well_startdate(df)
    check_total_number_of_entries_per_plate(df)
    check_num_frames(df)
    check_all_plates_have_WT(df)
    check_non_null_num_mutations(df)


def main():
    mutations_df = construct_mutations_dataframe()
    identity_df = construct_identity_dataframe(mutations_df)

    plate_data = construct_plate_info_df()
    well_data = construct_well_info_df()
    exptl_data = merge_plate_and_well_info_dfs(well_data, plate_data)

    total_df = merge_identity_and_experimental_dfs(exptl_data, identity_df)

    final_df_sanity_checks(total_df)
    save_df_to_parquet(total_df)

    gene_descriptions = construct_gene_description_dataframe()
    write_dataframe(gene_descriptions, f"gene_descriptions.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG if DEV_MODE else logging.INFO)
    main()
