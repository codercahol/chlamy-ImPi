import itertools
import logging
from typing import Callable

import numpy as np
from skimage import morphology
from skimage.morphology import binary_opening

logger = logging.getLogger(__name__)


# The following group of functions are all possible ways to generate an array of masks, given an image array


def compute_threshold_mask(
    img_arr: np.array,
    num_std: float = 3,
    use_opening: bool = False,
    time_reduction_fn: Callable = np.min,
    return_threshold: bool = False,
) -> np.array:
    """Function to compute a threshold-based mask array (i.e. masks all wells in a single plate)"""
    assert len(img_arr.shape) == 5

    threshold = compute_std_threshold(img_arr, num_std)
    img_arr_alltime = time_reduction_fn(img_arr, axis=2)

    assert len(img_arr_alltime.shape) == 4
    assert img_arr_alltime.shape[2] == img_arr.shape[3]
    assert img_arr_alltime.shape[3] == img_arr.shape[4]

    mask_arr = img_arr_alltime > threshold

    if use_opening:
        for i, j in itertools.product(range(img_arr.shape[0]), range(img_arr.shape[1])):
            mask_arr[i, j] = binary_opening(mask_arr[i, j])

    if return_threshold:
        return mask_arr, threshold
    else:
        return mask_arr


def compute_std_threshold(img_arr, num_std: float = 3.0):
    """Compute the background threshold, designed to be robust to case where there is not a blank in the top left"""
    assert len(img_arr.shape) == 5

    # First determine if top left well is indeed blank
    global_avg = np.mean(img_arr)
    global_std = np.std(img_arr)
    top_left_avg = np.mean(img_arr[0, 0])

    if abs(global_avg - top_left_avg) / global_std > 3.0:
        # We think that the top left cell is not blank
        logger.debug(f"Top left cell not blank")
        # Fall back to median of all wells, assumes that the bright spots form a minority
        threshold = np.median(img_arr)
    else:
        threshold = top_left_avg + num_std * np.std(img_arr[0, 0])

    return threshold


# The next set of functions are used to assess various properties of the masks


def count_empty_wells(mask_array):
    """
    Estimate the error due to misplating, which results in wells with no growing cells.

    Input:
        mask_array: 4D numpy array of shape (num_rows, num_columns, height, width)
    """
    mask_array_flat_im = mask_array.reshape(mask_array.shape[:2] + (-1,))
    total_wells = mask_array.shape[0] * mask_array.shape[1]
    num_good_wells = np.sum(np.max(mask_array_flat_im, axis=-1))
    empty_wells = total_wells - num_good_wells
    return empty_wells


def average_mask_area(mask_array) -> tuple[float, float]:
    sizes = np.sum(np.sum(mask_array, axis=-1), axis=-1)
    return np.mean(sizes), np.std(sizes)


def count_overlapping_masks(mask_array) -> int:
    """Perform some checks on the well masks.

    Returns the number of well masks which overlap with the boundary of the sub-image of the well.
    """
    array_shape = mask_array.shape

    num_overlapping = 0
    for i, j in itertools.product(range(array_shape[0]), range(array_shape[1])):
        if has_true_on_boundary(mask_array[i, j]):
            num_overlapping += 1

    logger.debug(f"We have found overlapping masks for {num_overlapping} masks")

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
