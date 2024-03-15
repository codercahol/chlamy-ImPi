# Authors: Murray Cutforth, Leron Perez
# Date: December 2023
#
# This script is given a directory of .tiff images, each one containing multiple time points from a single plate
# It will automatically divide the wells in each image, and compute a mean fluorescence value for each well
# at each time point. It will then write out this array (shape = (timepoints, rows, columns)) to a new .npy file.
# The input and output directories are specified in chlamy_impi/paths.py
# %%
from loguru import logger

import numpy as np
from skimage import io
from segment_multiwell_plate import segment_multiwell_plate
from tqdm import tqdm

from chlamy_impi.database_creation.error_correction import (
    remove_failed_photos,
    remove_repeated_initial_frame_tif,
)
from chlamy_impi.lib.visualize_well_segmentation import (
    visualise_channels,
    visualise_well_histograms,
    visualise_grid_crop,
)
from chlamy_impi.paths import (
    find_all_tif_images,
    well_segmentation_output_dir_path,
    npy_img_array_path,
    validate_inputs,
    well_segmentation_visualisation_dir_path,
    well_segmentation_histogram_dir_path,
)


OUTPUT_VISUALISATIONS = True
LOGGING_LEVEL = logging.DEBUG


def load_image(filename):
    return io.imread(filename)  # open tiff file in read mode


def save_img_array(img_array, name):

    outdir = npy_img_array_path(name).parent
    if not outdir.exists():
        outdir.mkdir(parents=True)

    logger.info(
        f"Saving image array of shape {img_array.shape} to {npy_img_array_path(name)}"
    )
    np.save(npy_img_array_path(name), img_array.astype(np.float32))


def main(rewrite=False):
    logger.info("\n" + "=" * 32 + "\nStarting main.py...\n" + "=" * 32)
    validate_inputs()
    filenames = find_all_tif_images()

    sep = "\n\t"
    logger.info(
        f"Found a total of {len(filenames)} tif files: \n\t{sep.join(str(x) for x in filenames)}"
    )

    failed_files = []

    for filename in tqdm(filenames):
        name = filename.stem

        if not rewrite and npy_img_array_path(name).exists():
            logger.info(f"Skipping {name} as it already exists")
            continue

        well_segmentation_output_dir_path(name).mkdir(parents=True, exist_ok=True)

        try:
            tif = load_image(filename)

            logger.debug(f"NUM_TIMESTEPS={tif.shape[0]}")

            tif = remove_failed_photos(tif)

            tif = remove_repeated_initial_frame_tif(tif)

            img_array, well_coords, i_vals, j_vals = segment_multiwell_plate(
                tif,
                peak_finder_kwargs={"peak_prominence": 1 / 25, "filter_threshold": 0.2},
                blob_log_kwargs={"threshold": 0.12},
                output_full=True,
            )

            # We expect all plates to be 16x24 in our study
            assert img_array.shape[0] == 16
            assert img_array.shape[1] == 24

            save_img_array(img_array, name)

            if OUTPUT_VISUALISATIONS:
                visualise_channels(
                    tif, savedir=well_segmentation_visualisation_dir_path(name)
                )
                visualise_well_histograms(
                    img_array, name, savedir=well_segmentation_histogram_dir_path(name)
                )
                visualise_grid_crop(
                    tif,
                    img_array,
                    i_vals,
                    j_vals,
                    well_coords,
                    savedir=well_segmentation_visualisation_dir_path(name),
                )

        except AssertionError as e:
            logger.error(f"File: {filename.stem} failed an assertion: {e}")
            failed_files.append((filename.stem, e))

    logger.info(f"Failed on {len(failed_files)} tif files: {failed_files}")

    logger.info("Program completed normally")


if __name__ == "__main__":
    logging.basicConfig(level=LOGGING_LEVEL)
    main()


# %%
