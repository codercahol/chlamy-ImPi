# Authors: Murray Cutforth, Leron Perez
# Date: December 2023
#
# This script is given a directory of .tiff images, each one containing multiple time points from a single plate
# It will automatically divide the wells in each image, and compute a mean fluorescence value for each well
# at each time point. It will then write out this array (shape = (timepoints, rows, columns)) to a new .npy file.

import itertools
import multiprocessing
from pathlib import Path
import logging
from multiprocessing import Pool

import numpy as np
from scipy import ndimage
from skimage import io, morphology, filters
import matplotlib.pyplot as plt
from tqdm import tqdm

from lib.constants import *

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


# For now, input and output dir hard-coded here
INPUT_DIR = Path("../data")
OUTPUT_DIR = Path("../output/image_processing/v1")
USE_MULTIPROCESSING = False


def validate_inputs():
    assert INPUT_DIR.exists()
    assert len(list(INPUT_DIR.glob("*.tif"))) > 0


def find_all_images():
    return list(INPUT_DIR.glob("*.tif"))


def load_image(filename):
    return io.imread(filename)  # open tiff file in read mode


def visualise_channels(tif, savedir, max_channels=5):
    logger.debug(f"Writing out plots of all time points in {savedir}")

    savedir.mkdir(parents=True, exist_ok=True)

    shape = tif.shape

    for channel in range(min(shape[0], max_channels)):
        fig, ax = plt.subplots(1, 1)
        ax.imshow(tif[channel, :, :])
        fig.savefig(savedir / f"{channel}.png")
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

    logger.info(f"Discarding {sum(~keep_image)} images (indices {list(np.argwhere(~keep_image))}) which are all black")

    tif = tif[keep_image]
    return tif


def generate_grid_crop_coordinates_hardcoded(x_min, x_max, y_min, y_max, num_rows, num_cols):
    """
    Generate the coordinates for the grid crop
    """
    # args are (start, end, num_points)
    # +1 is added to the number of points to include the end point
    # (eg. we want the number of fenceposts, not the number of fences)
    assert (x_max - x_min) % num_rows == 0, f"{x_max - x_min} is not divisible by {num_rows}"
    assert (y_max - y_min) % num_cols == 0

    i_vals = np.linspace(x_min, x_max, num_rows + 1, dtype=int)
    j_vals = np.linspace(y_min, y_max, num_cols + 1, dtype=int)
    return i_vals, j_vals


def resample_2d_image(tif_2d, iv, jv, subcell_resolution):
    order = 3 if USE_MULTIPROCESSING else 1
    subcell_img = ndimage.map_coordinates(tif_2d, [iv.ravel(), jv.ravel()], order=order)
    subcell_img = subcell_img.reshape((subcell_resolution, subcell_resolution))
    return subcell_img


def grid_crop(tif, i_vals, j_vals, subcell_resolution=20) -> np.array:
    """Crop and resample the image to each well
    """

    logger.debug("Starting grid_crop...")

    for i, i_next in itertools.pairwise(i_vals):
        assert abs(i_next - i - i_vals[1] + i_vals[0]) < np.finfo(np.float32).eps, "Unequal height of grid cells"
    for j, j_next in itertools.pairwise(j_vals):
        assert abs(j_next - j - j_vals[1] + j_vals[0]) < np.finfo(np.float32).eps, "Unequal width of grid cells"

    img_shape = tif.shape
    grid_shape = (len(i_vals), len(j_vals))

    # Prepare storage for resampled well images
    result = np.zeros(shape=(grid_shape[0] - 1, grid_shape[1] - 1, img_shape[0], subcell_resolution, subcell_resolution),
                      dtype=tif.dtype)

    for i, j in itertools.product(range(grid_shape[0] - 1), range(grid_shape[1] - 1)):
        # We want to resample a subcell_res x subcell_res image from the cell centred over well (i,j)
        i_start, i_end, j_start, j_end = i_vals[i], i_vals[i+1], j_vals[j], j_vals[j+1]

        # logger.debug(f"Resampling image for well {(i,j)}, using image coordinates {i_start}-{i_end}, {j_start}-{j_end}")

        iv, jv = np.meshgrid(np.linspace(i_start, i_end, subcell_resolution, dtype=np.float32),
                             np.linspace(j_start, j_end, subcell_resolution, dtype=np.float32),
                             indexing="ij")

        resample_args = [(tif[i].copy(), iv.copy(), jv.copy(), subcell_resolution) for i in range(len(tif))]

        # This is slow if we use 3rd order interpolation so I've parallelised it
        if USE_MULTIPROCESSING:
            with multiprocessing.Pool(multiprocessing.cpu_count() // 2) as p:
                subcell_images = list(p.starmap(resample_2d_image, resample_args))
        else:
            subcell_images = list(itertools.starmap(resample_2d_image, resample_args))

        result[i, j] = np.array(subcell_images)

    return result  # result[i, j, tstep] return cropped and resampled image at position (i,j) of grid


def visualise_grid_crop(tif, img_array, i_vals, j_vals, savedir, max_channels=5):
    logger.debug(f"Writing out plots of grid crop in {savedir}")
    savedir.mkdir(parents=True, exist_ok=True)

    img_shape = tif.shape
    array_shape = img_array.shape

    iv, jv = np.meshgrid(i_vals, j_vals, indexing="ij")

    for channel in range(min(img_shape[0], max_channels)):
        fig, ax = plt.subplots(1, 1)
        ax.imshow(tif[channel, :, :])
        ax.scatter(jv, iv, color="red", marker="o", s=3)
        fig.savefig(savedir / f"{channel}_gridvertices.png")
        plt.close(fig)

        fig, axs = plt.subplots(array_shape[0], array_shape[1])
        for i, j in itertools.product(range(array_shape[0]), range(array_shape[1])):
            ax = axs[i, j]
            ax.axis("off")
            ax.imshow(img_array[i, j, channel], vmin=tif[channel].min(), vmax=tif[channel].max())
        fig.savefig(savedir / f"{channel}_array.png")
        plt.close(fig)


def estimate_noise_threshold(img_array):
    mean = img_array[0, 0, :].mean()
    std = img_array[0, 0, :].std()
    threshold = mean + 3 * std

    logger.info(f"Computed threshold using blank control. mean : {mean}, std {std}, threshold {threshold}")

    return threshold


def find_mask_array(img_array, threshold):
    # Minimum value of a pixel across all timesteps must be above threshold
    # Return array of shape [nx, ny, height, width]
    img_array_mins = np.min(img_array, axis=2)
    mask_array = img_array_mins > threshold

    disk_radius = min(img_array.shape[-2:]) // 2
    disk_mask = morphology.disk(radius=disk_radius, dtype=bool)

    if disk_mask.shape[0] > img_array.shape[3]:
        assert img_array.shape[3] == img_array.shape[4]  # Check assumption of square cells
        disk_mask = disk_mask[:-1, :-1]

    assert disk_mask.shape == img_array.shape[-2:]

    mask_array = np.logical_and(mask_array, disk_mask)

    return mask_array


def visualise_mask_array(mask_array, savedir):
    logger.debug(f"Writing out plot of masks to {savedir}")
    savedir.mkdir(parents=True, exist_ok=True)

    array_shape = mask_array.shape

    fig, axs = plt.subplots(array_shape[0], array_shape[1])
    for i, j in itertools.product(range(array_shape[0]), range(array_shape[1])):
        ax = axs[i, j]
        ax.axis("off")
        ax.imshow(mask_array[i, j])
    fig.savefig(savedir / f"mask_array.png")
    plt.close(fig)


def visualise_mean_fluor(mean_fluor_array, savedir):
    logger.debug(f"Writing out time series of mean fluoro in {savedir}")
    savedir.mkdir(parents=True, exist_ok=True)

    array_shape = mean_fluor_array.shape
    xs = list(range(array_shape[2]))

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    for i, j in itertools.product(range(array_shape[0]), range(array_shape[1])):
        ax.plot(xs[::2], mean_fluor_array[i, j, ::2], color="black", linewidth=1)

    fig.savefig(savedir / "mean_fluor.png")
    plt.close(fig)


def misplating_error(mask_array, num_blanks):
    """
    Estimate the error due to misplating, which results in wells with no growing cells.

    Input:
        mask_array: 4D numpy array of shape (num_rows, num_columns, height, width)
        num_blanks: number of blanks in the plate
    """
    mask_array_flat_im = mask_array.reshape(mask_array.shape[:2] + (-1,))
    total_wells = mask_array.shape[0] * mask_array.shape[1] - num_blanks
    num_good_wells = np.sum(np.max(mask_array_flat_im, axis=-1)) - num_blanks
    misplated_error = (total_wells - num_good_wells) / total_wells
    return misplated_error


def save_mean_array(mean_fluor_array, name):
    outfile = OUTPUT_DIR / "mean_arrays" / f"{name}.npy"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    np.save(outfile, mean_fluor_array)
    logger.info(f"Mean fluorescence array saved out to: {outfile}")


def process_single_image(filename):
    name = filename.stem
    tif = load_image(filename)

    visualise_channels(tif, savedir=OUTPUT_DIR / name / "raw")

    tif_blurred = filters.gaussian(tif, sigma=1, channel_axis=0)

    visualise_channels(tif_blurred, savedir=OUTPUT_DIR / name / "blurred")

    tif = remove_failed_photos(tif)

    # TODO - check up on this
    # should be the number of images divided by 2::Int bc we have F0 and Fm
    NUM_TIMESTEPS = tif.shape[0]

    logger.debug(f"NUM_TIMESTEPS={NUM_TIMESTEPS}")

    # TODO: create more robust method to get coordinates of lattice - currently hardcoded to certain pixels
    i_vals, j_vals = generate_grid_crop_coordinates_hardcoded(
        x_min=X_MIN,
        x_max=X_MAX,
        y_min=Y_MIN,
        y_max=Y_MAX,
        num_rows=NUM_ROWS,
        num_cols=NUM_COLUMNS
    )

    img_array = grid_crop(tif, i_vals, j_vals)

    visualise_grid_crop(tif, img_array, i_vals, j_vals, savedir=OUTPUT_DIR / name / "grid")

    threshold = estimate_noise_threshold(img_array)

    mask_array = find_mask_array(img_array, threshold)

    logger.info(f"Misplating error: {misplating_error(mask_array, 7)}")

    visualise_mask_array(mask_array, savedir=OUTPUT_DIR / name / "masks")

    masked_img_array = img_array * np.expand_dims(mask_array, axis=2)

    mean_fluor_array = np.mean(np.mean(masked_img_array, axis=4), axis=3)

    visualise_mean_fluor(mean_fluor_array, savedir=OUTPUT_DIR / name / "mean_time_series")

    save_mean_array(mean_fluor_array, name)


def main():
    logger.info("\n" + "=" * 32 + "\nStarting image_processing.py...\n" + "=" * 32)
    validate_inputs()
    filenames = find_all_images()

    sep = "\n\t"
    logger.info(f"Found a total of {len(filenames)} tif files: \n\t{sep.join(str(x) for x in filenames)}")

    logger.debug(f"NUM_ROWS={NUM_ROWS}")
    logger.debug(f"NUM_COLS={NUM_COLUMNS}")
    logger.debug(f"width=({CELL_WIDTH_X}, {CELL_WIDTH_Y})")
    logger.debug(f"X_MIN={X_MIN}")
    logger.debug(f"X_MAX={X_MAX}")
    logger.debug(f"Y_MIN={Y_MIN}")
    logger.debug(f"Y_MAX={Y_MAX}")

    for filename in tqdm(filenames):
        process_single_image(filename)

    logger.info("Program completed normally")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()


    # TODO: future method might do the following
    # 0.5: average image over all timepoints?
    # 1. Use Laplacian of gaussian method to place point on centre of each well
    # 2. Rotate image so that grid is aligned with image space
    # For this step, bin the coordinates and optimise for sparsity
    # Exhaustively try all angles of rotation
    # For objective function, we could create histograms of x and y coords, then interpret as probability distribution and choose the one with minimum entropy
    # We could also look at the Fourier transform of this histogram
    # 3. Find cell centres on x and y dimensions
    # Need to specify the number of cells, start position, and cell width
    # After projecting onto each axis, count number of peaks, then do a linear least squares to minimise L2 error of grid
    # 4. Convert cell centres to cell edges, round to nearest integer
