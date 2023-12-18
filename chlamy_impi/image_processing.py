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
import matplotlib.pyplot as plt
import matplotlib
from segment_multiwell_plate import segment_multiwell_plate
from tqdm import tqdm
import pandas as pd
import platform

# Fix for matplotlib memory leak, see https://github.com/matplotlib/matplotlib/issues/20067
if platform.system() == "Linux":
    matplotlib.use("agg")

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


# For now, input and output dir hard-coded here
INPUT_DIR = Path("./data")
OUTPUT_DIR = Path("./output/image_processing/v3")
USE_MULTIPROCESSING = False
OUTPUT_VISUALISATIONS = True
LOGGING_LEVEL = logging.DEBUG


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


def visualise_channels(tif, savedir, max_channels=None):
    logger.debug(f"Writing out plots of all time points in {savedir}")

    savedir.mkdir(parents=True, exist_ok=True)

    shape = tif.shape

    if max_channels is None:
        max_channels = shape[0]

    for channel in range(min(shape[0], max_channels)):
        fig, ax = plt.subplots(1, 1)
        ax.imshow(tif[channel, :, :])
        fig.savefig(savedir / f"{channel}.png")
        fig.clf()
        plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(np.mean(tif, axis=0))
    fig.savefig(savedir / f"avg.png")
    fig.clf()
    plt.close(fig)


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

    logger.info(
        f"Discarding {sum(~keep_image)} images (indices {list(np.argwhere(~keep_image))}) which are all black"
    )

    tif = tif[keep_image]
    return tif, sum(~keep_image)


def visualise_grid_crop(
    tif, img_array, i_vals, j_vals, well_coords, savedir, max_channels=5
):
    logger.debug(f"Writing out plots of grid crop in {savedir}")
    savedir.mkdir(parents=True, exist_ok=True)

    img_shape = tif.shape
    array_shape = img_array.shape

    iv, jv = np.meshgrid(i_vals, j_vals, indexing="ij")
    iv2, jv2 = np.meshgrid(i_vals, j_vals, indexing="xy")

    for channel in range(min(img_shape[0], max_channels)):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(tif[channel, :, :])
        # Draw well centre coords
        ax.scatter(
            list(zip(*well_coords))[1],
            list(zip(*well_coords))[0],
            color="red",
            marker="x",
            s=2,
        )
        # Draw grid
        ax.plot(jv, iv, color="red")
        ax.plot(jv2, iv2, color="red")
        fig.savefig(savedir / f"{channel}_grid.png")
        fig.clf()
        plt.close(fig)

        fig, axs = plt.subplots(array_shape[0], array_shape[1])
        for i, j in itertools.product(range(array_shape[0]), range(array_shape[1])):
            ax = axs[i, j]
            ax.axis("off")
            ax.imshow(
                img_array[i, j, channel],
                vmin=tif[channel].min(),
                vmax=tif[channel].max(),
            )
        fig.savefig(savedir / f"{channel}_subimage_array.png")
        fig.clf()
        plt.close(fig)


def estimate_noise_threshold(img_array):
    mean = img_array[0, 0, :].mean()
    std = img_array[0, 0, :].std()
    threshold = mean + 3 * std

    logger.info(
        f"Computed threshold using blank control. mean : {mean}, std {std}, threshold {threshold}"
    )

    return threshold


def find_mask_array(img_array, threshold):
    # Minimum value of a pixel across all timesteps must be above threshold
    # Return array of shape [nx, ny, height, width]
    img_array_mins = np.min(img_array, axis=2)
    mask_array = img_array_mins > threshold

    # disk_mask = get_disk_mask(img_array)
    # mask_array = np.logical_and(mask_array, disk_mask)

    return mask_array


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

    # assert num_overlapping <= 3, f"We have found overlapping masks for {num_overlapping} masks"
    logger.info(f"We have found overlapping masks for {num_overlapping} masks")

    return num_overlapping


def has_true_on_boundary(arr):
    """Check if mask reaches edge of cell - should always be false"""

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
        assert (
            img_array.shape[3] == img_array.shape[4]
        )  # Check assumption of square cells
        disk_mask = disk_mask[:-1, :-1]
    assert disk_mask.shape == img_array.shape[-2:]
    return disk_mask


def visualise_mask_array(mask_array, savedir):
    logger.debug(f"Writing out plot of masks to {savedir}")
    savedir.mkdir(parents=True, exist_ok=True)

    array_shape = mask_array.shape

    fig, axs = plt.subplots(array_shape[0], array_shape[1])
    for i, j in itertools.product(range(array_shape[0]), range(array_shape[1])):
        ax = axs[i, j]
        ax.axis("off")
        ax.imshow(mask_array[i, j])
    fig.savefig(savedir / "mask_array.png")
    fig.clf()
    plt.close(fig)


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


def save_mean_array(mean_fluor_array, name):
    outfile = OUTPUT_DIR / "mean_arrays" / f"{name}.npy"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    np.save(outfile, mean_fluor_array)
    logger.info(f"Mean fluorescence array saved out to: {outfile}")

    # Also save as csv to be read using pandas
    rows = []
    for i, j in itertools.product(
        range(mean_fluor_array.shape[0]), range(mean_fluor_array.shape[1])
    ):
        col_to_val = {
            f"mean_fluorescence_frame_{k}": mean_fluor_array[i, j, k]
            for k in range(mean_fluor_array.shape[2])
        }
        col_to_val.update({"row": i, "col": j})
        rows.append(col_to_val)
    df = pd.DataFrame(rows)
    outfile = OUTPUT_DIR / "mean_arrays" / f"{name}.csv"
    df.to_csv(outfile)
    logger.info(f"Mean fluorescence array saved out to: {outfile}")


def process_single_image(filename):
    return


# TODO: create functions to get path to csv file with mean fluor values, and path to per-file results folder


def main():
    logger.info("\n" + "=" * 32 + "\nStarting image_processing.py...\n" + "=" * 32)
    validate_inputs()
    filenames = find_all_images()

    sep = "\n\t"
    logger.info(
        f"Found a total of {len(filenames)} tif files: \n\t{sep.join(str(x) for x in filenames)}"
    )

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
                tif, peak_finder_kwargs={"peak_prominence": 0.1}, output_full=True
            )

            threshold = estimate_noise_threshold(img_array)

            logger.info(f"Threshold = {threshold}")

            mask_array = find_mask_array(img_array, threshold)

            empty_wells, total_wells = count_empty_wells(mask_array)

            logger.info(f"Found a total of {empty_wells} / {total_wells} empty wells")

            num_overlapping = validate_well_mask_array(mask_array)

            masked_img_array = img_array * np.expand_dims(mask_array, axis=2)

            mean_fluor_array = np.mean(np.mean(masked_img_array, axis=4), axis=3)

            save_mean_array(mean_fluor_array, name)

            plate_info = {
                "name": name,
                "timepoints": tif.shape[0],
                "num_blank_frames": num_blank_frames,
                "threshold": threshold,
                "total_wells": total_wells,
                "num_empty_wells": empty_wells,
                "num_overlapping_masks": num_overlapping,
            }

            df = load_plate_info()
            new_row = pd.DataFrame([plate_info])
            df = pd.concat([df, new_row], ignore_index=True)
            write_plate_info(df)

            if OUTPUT_VISUALISATIONS:
                visualise_channels(tif, savedir=results_dir_path(name) / "raw")
                visualise_mask_array(
                    mask_array, savedir=results_dir_path(name) / "masks"
                )
                visualise_grid_crop(
                    tif,
                    img_array,
                    i_vals,
                    j_vals,
                    well_coords,
                    savedir=results_dir_path(name) / "grid",
                )

        except AssertionError as e:
            logger.error(f"File: {filename.stem}. Error: {e}")
            failed_files.append((filename.stem, e))

    logger.info(f"Failed on {len(failed_files)} tif files: {failed_files}")

    logger.info("Program completed normally")


if __name__ == "__main__":
    logging.basicConfig(level=LOGGING_LEVEL)
    main()
