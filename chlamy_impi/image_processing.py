# Authors: Murray Cutforth, Leron Perez
# Date: December 2023
#
# This script is given a directory of .tiff images, each one containing multiple time points from a single plate
# It will automatically divide the wells in each image, and compute a mean fluorescence value for each well
# at each time point. It will then write out this array (shape = (timepoints, rows, columns)) to a new .npy file.
import itertools
from pathlib import Path
import logging

import numpy as np
from skimage import io, morphology
import matplotlib
from segment_multiwell_plate import segment_multiwell_plate
from tqdm import tqdm
import pandas as pd

from chlamy_impi.lib.visualize import visualise_channels, visualise_grid_crop, visualise_mask_array, visualise_well_histograms

logger = logging.getLogger(__name__)
matplotlib.use('agg')  # Fix for matplotlib memory leak, see https://github.com/matplotlib/matplotlib/issues/20067
logging.getLogger("matplotlib").setLevel(logging.WARNING)


# For now, input and output dir hard-coded here
INPUT_DIR = Path("../data")
OUTPUT_DIR = Path("../output/image_processing/v6")
USE_MULTIPROCESSING = False
OUTPUT_VISUALISATIONS = True
LOGGING_LEVEL = logging.DEBUG


# TODO: for each plate, plot intensity histograms of each well, to inform thresholding
# TODO: collect global intensity histogram stats


def validate_inputs():
    assert INPUT_DIR.exists()
    assert len(list(INPUT_DIR.glob("*.tif"))) > 0


def find_all_images():
    return list(INPUT_DIR.glob("*.tif"))


def load_image(filename):
    return io.imread(filename)  # open tiff file in read mode


def results_dir_path(name) -> Path:
    savedir = OUTPUT_DIR / name
    return savedir


def plate_info_path():
    return OUTPUT_DIR / "plate_info.csv"


def load_plate_info() -> pd.DataFrame:
    try:
        df = pd.read_csv(plate_info_path(), index_col=0, header=0)
        return df
    except FileNotFoundError:
        return pd.DataFrame()


def write_plate_info(df: pd.DataFrame) -> None:
    df.to_csv(plate_info_path())


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


def estimate_noise_threshold(img_array):
    mean = img_array[0, 0, :].mean()
    std = img_array[0, 0, :].std()
    threshold = mean + 3 * std

    logger.info(f"Computed threshold using blank control. mean : {mean}, std {std}, threshold {threshold}")

    return threshold


#def find_mask_array(img_array, threshold):
#    # Minimum value of a pixel across all timesteps must be above threshold
#    # Return array of shape [nx, ny, height, width]
#    img_array_mins = np.min(img_array, axis=2)
#    mask_array = img_array_mins > threshold
#
#    #disk_mask = get_disk_mask(img_array)
#    #mask_array = np.logical_and(mask_array, disk_mask)
#
#    return mask_array


def validate_well_mask_array(mask_array) -> int:
    """Perform some checks on the well masks.

    Returns the number of well masks which overlap with the boundary of the sub-image of the well.
    """
    array_shape = mask_array.shape
    arr = np.zeros_like(mask_array[:, :, 0, 0])

    for i, j in itertools.product(range(array_shape[0]), range(array_shape[1])):
        if has_true_on_boundary(mask_array[i, j]):
            logger.warning(f"Mask {i},{j} has hit boundary")
            arr[i, j] = True

    num_overlapping = np.sum(arr)

    #assert num_overlapping <= 3, f"We have found overlapping masks for {num_overlapping} masks"
    logger.info(f"We have found overlapping masks for {num_overlapping} masks")

    return num_overlapping


def has_true_on_boundary(arr):
    """Check if mask reaches edge of cell - should always be false
    """

    # Check the top and bottom rows
    if np.any(arr[0, :]) or np.any(arr[-1, :]):
        return True

    # Check the left and right columns
    if np.any(arr[:, 0]) or np.any(arr[:, -1]):
        return True

    return False


def get_disk_mask(img_array):
    disk_radius = min(img_array.shape[-2:]) // 2
    disk_mask = morphology.disk(radius=disk_radius, dtype=bool)
    if disk_mask.shape[0] > img_array.shape[3]:
        assert img_array.shape[3] == img_array.shape[4]  # Check assumption of square cells
        disk_mask = disk_mask[:-1, :-1]
    assert disk_mask.shape == img_array.shape[-2:]
    return disk_mask


def count_empty_wells(mask_array):
    """
    Estimate the error due to misplating, which results in wells with no growing cells.

    Input:
        mask_array: 4D numpy array of shape (num_rows, num_columns, height, width)
        num_blanks: number of blanks in the plate
    """
    mask_array_flat_im = mask_array.reshape(mask_array.shape[:2] + (-1,))
    total_wells = mask_array.shape[0] * mask_array.shape[1]
    num_good_wells = np.sum(np.max(mask_array_flat_im, axis=-1))
    empty_wells = total_wells - num_good_wells
    return empty_wells, total_wells


def img_array_outpath(outdir, name):
    return outdir / f"{name}.npy"


def save_img_array(img_array, outdir, name):
    if not outdir.exists():
        outdir.mkdir(parents=True)
    np.save(img_array_outpath(outdir, name), img_array.astype(np.float32))


def load_img_array(outdir, name):
    return np.load(img_array_outpath(outdir, name))




def save_mean_array(mean_fluor_array, name):
    outfile = OUTPUT_DIR / "mean_arrays" / f"{name}.npy"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    np.save(outfile, mean_fluor_array)
    logger.info(f"Mean fluorescence array saved out to: {outfile}")

    # Also save as csv to be read using pandas
    rows = []
    for i, j in itertools.product(range(mean_fluor_array.shape[0]), range(mean_fluor_array.shape[1])):
        col_to_val = {f"mean_fluorescence_frame_{k}": mean_fluor_array[i, j, k] for k in range(mean_fluor_array.shape[2])}
        col_to_val.update({"row": i, "col": j})
        rows.append(col_to_val)
    df = pd.DataFrame(rows)
    outfile = OUTPUT_DIR / "mean_arrays" / f"{name}.csv"
    df.to_csv(outfile)
    logger.info(f"Mean fluorescence array saved out to: {outfile}")


def main():
    logger.info("\n" + "=" * 32 + "\nStarting image_processing.py...\n" + "=" * 32)
    validate_inputs()
    filenames = find_all_images()

    sep = "\n\t"
    logger.info(f"Found a total of {len(filenames)} tif files: \n\t{sep.join(str(x) for x in filenames)}")

    failed_files = []

    for filename in tqdm(filenames):
        name = filename.stem

        if results_dir_path(name).exists():
            logger.info(f"Image {name} has already been processed. Skipping.")
            continue
        else:
            results_dir_path(name).mkdir(parents=True)

        try:
            tif = load_image(filename)

            logger.debug(f"NUM_TIMESTEPS={tif.shape[0]}")

            tif, num_blank_frames = remove_failed_photos(tif)

            img_array, well_coords, i_vals, j_vals = segment_multiwell_plate(
                tif,
                peak_finder_kwargs={"peak_prominence": 1 / 25, "filter_threshold": 0.2},
                blob_log_kwargs={"threshold": 0.12},
                output_full=True)

            # We expect all plates to be 16x24
            assert img_array.shape[0] == 16
            assert img_array.shape[1] == 24

            save_img_array(img_array, OUTPUT_DIR / "img_array", name)

            #threshold = estimate_noise_threshold(img_array)

            #logger.info(f"Threshold = {threshold}")

            #mask_array = find_mask_array(img_array, threshold)

            #empty_wells, total_wells = count_empty_wells(mask_array)

            #logger.info(f"Found a total of {empty_wells} / {total_wells} empty wells")

            #num_overlapping = validate_well_mask_array(mask_array)

            #masked_img_array = img_array * np.expand_dims(mask_array, axis=2)

            # TODO: is this invalid, we are including the zeroes in the mean computation?
            #mean_fluor_array = np.mean(np.mean(masked_img_array, axis=4), axis=3)

            #save_mean_array(mean_fluor_array, name)

            #plate_info = {"name": name,
            #     "timepoints": tif.shape[0],
            #     "num_blank_frames": num_blank_frames,
            #     "threshold": threshold,
            #     "total_wells": total_wells,
            #     "num_empty_wells": empty_wells,
            #     "num_overlapping_masks": num_overlapping}

            #df = load_plate_info()
            #new_row = pd.DataFrame([plate_info])
            #df = pd.concat([df, new_row], ignore_index=True)
            #write_plate_info(df)

            if OUTPUT_VISUALISATIONS:
                visualise_channels(tif, savedir=results_dir_path(name) / "raw")
                visualise_well_histograms(img_array, name, savedir=OUTPUT_DIR / "histograms")
                visualise_grid_crop(tif, img_array, i_vals, j_vals, well_coords, savedir=results_dir_path(name) / "grid")
            #    visualise_mask_array(mask_array, savedir=results_dir_path(name) / "masks")

        except AssertionError as e:
            logger.error(f"File: {filename.stem}. Error: {e}")
            failed_files.append((filename.stem, e))

    logger.info(f"Failed on {len(failed_files)} tif files: {failed_files}")

    logger.info("Program completed normally")


if __name__ == "__main__":
    logging.basicConfig(level=LOGGING_LEVEL)
    main()


