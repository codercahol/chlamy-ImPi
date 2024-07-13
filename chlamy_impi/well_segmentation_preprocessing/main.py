# Authors: Murray Cutforth, Leron Perez
#
# This script is given a directory of .tiff images, each one containing multiple time points from a single plate
# It will automatically divide the wells in each image, and compute a mean fluorescence value for each well
# at each time point. It will then write out this array (shape = (timepoints, rows, columns)) to a new .npy file.
# The input and output directories are specified in chlamy_impi/paths.py

import logging

import numpy as np
from segment_multiwell_plate.segment_multiwell_plate import correct_rotations
from skimage import io
from segment_multiwell_plate import segment_multiwell_plate, find_well_centres
from tqdm import tqdm

from chlamy_impi.database_creation.error_correction import remove_failed_photos, remove_repeated_initial_frame_tif
from chlamy_impi.database_creation.utils import parse_name
from chlamy_impi.lib.visualize_well_segmentation import visualise_channels, visualise_well_histograms, \
    visualise_grid_crop
from chlamy_impi.paths import find_all_tif_images, well_segmentation_output_dir_path, npy_img_array_path, \
    validate_inputs, well_segmentation_visualisation_dir_path, well_segmentation_histogram_dir_path
from chlamy_impi.well_segmentation_preprocessing.well_segmentation_assertions import assert_expected_shape

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


OUTPUT_VISUALISATIONS = True
LOGGING_LEVEL = logging.DEBUG


def load_image(filename):
    return io.imread(filename)  # open tiff file in read mode


def save_img_array(img_array, name):

    outdir = npy_img_array_path(name).parent
    if not outdir.exists():
        outdir.mkdir(parents=True)

    logger.info(f"Saving image array of shape {img_array.shape} to {npy_img_array_path(name)}")
    np.save(npy_img_array_path(name), img_array.astype(np.float32))


def main():
    logger.info("\n" + "=" * 32 + "\nStarting main.py...\n" + "=" * 32)
    validate_inputs()
    filenames = find_all_tif_images()

    sep = "\n\t"
    logger.info(f"Found a total of {len(filenames)} tif files: \n\t{sep.join(str(x) for x in filenames)}")

    for filename in tqdm(filenames):
        name = filename.stem
        outpath = well_segmentation_output_dir_path(name)
        plate_num, measurement_num, time_regime = parse_name(filename.name)

        logger.info(f"Processing plate_num={plate_num}, measurement_num={measurement_num}, time_regime={time_regime}")

        if outpath.exists():
            logger.info(f"Skipping {name} as it already exists")
            continue

        outpath.mkdir(parents=True, exist_ok=True)

        tif = load_image(filename)

        logger.debug(f"NUM_TIMESTEPS={tif.shape[0]}")

        tif = remove_failed_photos(tif)

        tif = remove_repeated_initial_frame_tif(tif)

        # Note - if these parameters need to be tuned, see test_rotation_correction.py
        img_array, well_coords, i_vals, j_vals = segment_multiwell_plate(
            tif,
            peak_finder_kwargs={"peak_prominence": 1 / 25, "filter_threshold": 0.2, "width": 2},
            blob_log_kwargs={"threshold": 0.12, "min_sigma": 1, "max_sigma": 2, "exclude_border": 10},
            output_full=True,
        )

        save_img_array(img_array, name)

        if OUTPUT_VISUALISATIONS:
            visualise_channels(tif, savedir=well_segmentation_visualisation_dir_path(name))
            visualise_well_histograms(img_array, name, savedir=well_segmentation_histogram_dir_path(name))
            visualise_grid_crop(tif, img_array, i_vals, j_vals, well_coords, savedir=well_segmentation_visualisation_dir_path(name))

        assert_expected_shape(i_vals, j_vals, plate_num)

    logger.info("Program completed normally")


def correct_image_rotation(im: np.array, well_coords: np.array, blob_log_kwargs: dict):
    """
    Correct the rotation of an image by finding the rotation angle and applying it to the image
    """
    rotated_im, rotation_angle = correct_rotations(im, well_coords, return_theta=True)

    # Iterate on the rotation correction, adding some random jitter on the well coordinates to smooth out errors in well centre estimation
    if rotation_angle > 0.01:
        print(f"Found rotation angle of {rotation_angle:.2g} rad. Iterating on correction")
        thetas = [rotation_angle]
        n_iters = 5

        for i in range(n_iters):
            well_coords_it = find_well_centres(rotated_im, **blob_log_kwargs)
            well_coords_it += np.random.normal(0, 0.5, well_coords_it.shape)
            rotated_im, rotation_angle = correct_rotations(rotated_im, well_coords_it, return_theta=True)
            thetas.append(rotation_angle)

        rotation_angle = np.sum(thetas)

    return rotated_im, rotation_angle


if __name__ == "__main__":
    logging.basicConfig(level=LOGGING_LEVEL)
    main()
