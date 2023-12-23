# Authors: Murray Cutforth, Leron Perez
# Date: December 2023
#
# This script is given a directory of .tiff images, each one containing multiple time points from a single plate
# It will automatically divide the wells in each image, and compute a mean fluorescence value for each well
# at each time point. It will then write out this array (shape = (timepoints, rows, columns)) to a new .npy file.
from pathlib import Path
import logging

import numpy as np
from skimage import io
import matplotlib
from segment_multiwell_plate import segment_multiwell_plate
from tqdm import tqdm

from chlamy_impi.lib.visualize_well_segmentation import visualise_channels, visualise_well_histograms, \
    visualise_grid_crop

logger = logging.getLogger(__name__)
matplotlib.use('agg')  # Fix for matplotlib memory leak, see https://github.com/matplotlib/matplotlib/issues/20067
logging.getLogger("matplotlib").setLevel(logging.WARNING)


# For now, input and output dir hard-coded here
INPUT_DIR = Path("../data")
OUTPUT_DIR = Path("../output/image_processing/v7")
USE_MULTIPROCESSING = False
OUTPUT_VISUALISATIONS = True
LOGGING_LEVEL = logging.DEBUG


def validate_inputs():
    assert INPUT_DIR.exists()
    assert len(list(INPUT_DIR.glob("*.tif"))) > 0
    assert len(list(INPUT_DIR.glob("*.tif"))) == len(set(INPUT_DIR.glob("*.tif")))


def find_all_images():
    return list(INPUT_DIR.glob("*.tif"))


def load_image(filename):
    return io.imread(filename)  # open tiff file in read mode


def results_dir_path(name) -> Path:
    savedir = OUTPUT_DIR / name
    return savedir


def remove_failed_photos(tif):
    """
    Remove photos that are all black

    Input:
        tif: torch tensor of shape (num_images, height, width)
    Output:
        tif: torch tensor of shape (num_images, height, width)
        photo_index: numpy array of indices of photos that were kept
    """
    max_per_timestep = tif.max(1).max(1)
    keep_image = max_per_timestep > 0

    logger.info(f"Discarding {sum(~keep_image)} images (indices {list(np.argwhere(~keep_image))}) which are all black")

    tif = tif[keep_image]
    return tif, sum(~keep_image)


def img_array_outpath(outdir, name):
    return outdir / f"{name}.npy"


def save_img_array(img_array, outdir, name):
    if not outdir.exists():
        outdir.mkdir(parents=True)

    logger.info(f"Saving image array of shape {img_array.shape} to {img_array_outpath(outdir, name)}")
    np.save(img_array_outpath(outdir, name), img_array.astype(np.float32))


def main():
    logger.info("\n" + "=" * 32 + "\nStarting run_well_segmentation_preprocessing.py...\n" + "=" * 32)
    validate_inputs()
    filenames = find_all_images()

    sep = "\n\t"
    logger.info(f"Found a total of {len(filenames)} tif files: \n\t{sep.join(str(x) for x in filenames)}")

    failed_files = []

    for filename in tqdm(filenames):
        name = filename.stem
        results_dir_path(name).mkdir(parents=True, exist_ok=True)

        try:
            tif = load_image(filename)

            logger.debug(f"NUM_TIMESTEPS={tif.shape[0]}")

            tif, num_blank_frames = remove_failed_photos(tif)

            img_array, well_coords, i_vals, j_vals = segment_multiwell_plate(
                tif,
                peak_finder_kwargs={"peak_prominence": 1 / 25, "filter_threshold": 0.2},
                blob_log_kwargs={"threshold": 0.12},
                output_full=True)

            # We expect all plates to be 16x24 in our study
            assert img_array.shape[0] == 16
            assert img_array.shape[1] == 24

            save_img_array(img_array, OUTPUT_DIR / "img_array", name)

            if OUTPUT_VISUALISATIONS:
                visualise_channels(tif, savedir=results_dir_path(name) / "raw")
                visualise_well_histograms(img_array, name, savedir=OUTPUT_DIR / "histograms")
                visualise_grid_crop(tif, img_array, i_vals, j_vals, well_coords, savedir=results_dir_path(name) / "grid")

        except AssertionError as e:
            logger.error(f"File: {filename.stem} failed an assertion: {e}")
            failed_files.append((filename.stem, e))

    logger.info(f"Failed on {len(failed_files)} tif files: {failed_files}")

    logger.info("Program completed normally")


if __name__ == "__main__":
    logging.basicConfig(level=LOGGING_LEVEL)
    main()


