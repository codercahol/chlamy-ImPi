import numpy as np

from chlamy_impi.lib.y2_functions import get_background_intensity


def compute_all_npq_averaged(img_array, mask_array) -> np.array:
    """Compute average NPQ for each well in an entire plate
    Returns a 3D numpy array of shape (Ni, Nj, num_steps)
    """
    Ni, Nj = img_array.shape[:2]

    # First subtract background light intensity from each time point
    backgrounds = get_background_intensity(img_array, mask_array)
    img_array = img_array - backgrounds[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]

    # TODO: smooth each 2D image?
    # TODO: refactor out common code from npq and y2 computation

    # Compute pixelwise NPQ values, for every pixel, ignoring mask
    Fm_array = img_array[:, :, 1:2, ...]
    Fm_prime_array = img_array[:, :, 3::2, ...]  # Skip Fm
    npq_array = (Fm_array - Fm_prime_array) / Fm_prime_array
    num_steps = Fm_prime_array.shape[2]
    assert num_steps == npq_array.shape[2]

    # Set pixels outside mask to nan, and take mean of non-nan pixels
    mask_array = np.broadcast_to(mask_array[:, :, np.newaxis, ...], (Ni, Nj, num_steps, mask_array.shape[2], mask_array.shape[-1]))
    npq_array[~mask_array] = np.nan
    npq_array_means = np.nanmean(npq_array.reshape(Ni, Nj, num_steps, -1), axis=-1)

    assert npq_array_means.shape == (Ni, Nj, num_steps)
    assert np.nanmax(npq_array_means) < 10.0
    assert np.nanmin(npq_array_means) > -0.1, np.nanmin(npq_array_means)

    return npq_array_means
