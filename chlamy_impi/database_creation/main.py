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

import pandas as pd
import numpy as np

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

    for filename_npy, filename_meta in zip(filenames_npy, filenames_meta):
        plate_num, measurement_num, light_regime = parse_name(filename_npy.name)

        logger.info(f"\n\n\nProcessing plate info from filename: {filename_npy.name}")

        img_array, _ = prepare_img_array_and_df(filename_meta, filename_npy)

        assert img_array.shape[2] % 2 == 0

        _, thresholds = compute_threshold_mask(img_array, return_thresholds=True)
        dark_threshold, light_threshold = thresholds

        row_data = {
            "plate": plate_num,
            "measurement": measurement_num,
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
    logger.info(f"{df.head()}")

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

    for filename_npy, filename_meta in zip(filenames_npy, filenames_meta):
        plate_num, measurement_num, light_regime = parse_name(filename_npy.name)

        logger.info(
            f"\n\n\nProcessing image features from filename: {filename_npy.name}"
        )

        try:
            img_array, meta_df = prepare_img_array_and_df(filename_meta, filename_npy)
        except Exception as e:
            logger.error(f"Error processing file {filename_npy.name}. Skipping.")
            logger.error(e)
            continue

        measurement_times = compute_measurement_times(meta_df=meta_df)

        assert img_array.shape[2] % 2 == 0
        assert img_array.shape[2] // 2 == len(measurement_times)


        mask_array = compute_threshold_mask(img_array)
        y2_array = compute_all_y2_averaged(img_array, mask_array)
        fv_fm_array = compute_all_fv_fm_averaged(img_array, mask_array)
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
    logger.info(f"{df.head()}")
    return df


def merge_plate_and_well_info_dfs(plate_df: pd.DataFrame, well_df: pd.DataFrame):
    """ merge the plate and well info dataframes """
    df = pd.merge(well_df, plate_df, on=["plate", "measurement"], how="left")
    
    df["well_id"] = df.apply(index_to_location_rowwise, axis=1)
        #lambda x: "{}-{}-{}".format(x.plate, x.measurement, index_to_location_rowwise(x)), axis=1)
    # df = df.set_index("full_id")
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
    logger.info(f"{df_gene_descriptions.head()}")
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
    logger.info(f"{df.head()}")
    return df


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

    # Assert that there are no null values in the "Location" and "New Location" columns
    assert df["Location"].notnull().all()
    assert df["New location"].notnull().all()

    # Collect columns which we need
    df = df.rename(columns={"New location": "plate", "Location": "well_id"})
    df = df[["mutant_ID", "plate", "well_id"]]
    df["plate"] = df["plate"].apply(spreadsheet_plate_to_numeric)
    df = df.drop_duplicates(ignore_index=True)

    # Add column which tells us the number of genes which were mutated
    signif_mutations = mutation_df[mutation_df.confidence_level <= conf_threshold]
    gene_mutation_counts = signif_mutations.groupby("mutant_ID").nunique()["gene"]

    mutated_genes = signif_mutations.groupby("mutant_ID").apply(
        lambda x: ",".join(set(x.gene))
    )
    mutated_genes = mutated_genes.reset_index().rename(columns={0: "mutated_genes"})
    df = pd.merge(df, mutated_genes, on="mutant_ID")
    
    df["num_mutations"] = df["mutant_ID"].apply(lambda x: gene_mutation_counts[x])

    # For each plate, add rows for the three WT wells
    for plate in df.plate.unique():
        for well_pos in get_wt_well_positions():
            # Check no well already exists at this location
            assert f"{plate}-{well_pos}" not in df.index
            WT_row = {
                "plate": plate,
                "well_id": well_pos,
                "mutant_ID": "WT",
                "mutated_genes": "",
                "num_mutations": 0,
            }
            df.loc[len(df)] = WT_row

    # Add rows for wild type plate 99
    wt_rows = create_wt_rows()
    df_wt = pd.DataFrame(wt_rows)
    df = pd.concat([df, df_wt], axis=0, ignore_index=False)

    # Perform some final sanity checks
    assert df["mutant_ID"].notnull().all()
    assert df["num_mutations"].notnull().all()
    assert df["num_mutations"].min() >= 0
    assert df["num_mutations"].max() <= 8

    # Group by the plate number (the first part of the index string)
    plates = df.plate
    plate_counts = plates.value_counts()
    for plate, count in plate_counts.items():
        assert count <= 384, f"Plate {plate} has {count} wells"

    logger.info(
        f"Constructed identity dataframe. Shape: {df.shape}. Columns: {df.columns}."
    )
    logger.info(f"Values of num_mutations: {df.num_mutations.unique()}")
    logger.info(f"{df.head()}")
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
    return ["C12", "N3", "N22"]


def well_position_iterator():
    for i in range(1, 17):
        for j in range(1, 25):
            yield f"{chr(ord('A') + i - 1)}{j}"


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
    non_blank_wells = exptl_data.well_id[exptl_data.well_id != "A1"]
    exptl_plate_n_well = set(product(exptl_data.plate, non_blank_wells))
    identity_plate_n_well = set(product(identity_df.plate, identity_df.well_id))

    err_msg = exptl_plate_n_well - identity_plate_n_well
    assert exptl_plate_n_well.issubset(identity_plate_n_well), err_msg

    total_df = pd.merge(exptl_data, identity_df, on=["plate", "well_id"], how="left")
    logger.info(f"Shape of total_df: {total_df.shape}, Columns: {total_df.columns}")
    logger.info(total_df.head())

    logger.info(f"After merge, we have data for plates: {total_df.plate.unique()}")
    logger.info(
        f"After merge, we have data for light regimes: {total_df.light_regime.unique()}"
    )
    logger.info(
        f"After merge, we have data for measurement numbers: {total_df.measurement.unique()}"
    )

    return total_df


def main():
    plate_data = construct_plate_info_df()
    well_data = construct_well_info_df()
    exptl_data = merge_plate_and_well_info_dfs(well_data, plate_data)
    
    mutations_df = construct_mutations_dataframe()
    identity_df = construct_identity_dataframe(mutations_df)

    total_df = merge_identity_and_experimental_dfs(exptl_data, identity_df)
    save_df_to_parquet(total_df)

    gene_descriptions = construct_gene_description_dataframe()
    write_dataframe(gene_descriptions, f"gene_descriptions.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG if DEV_MODE else logging.INFO)
    main()
