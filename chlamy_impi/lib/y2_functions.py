import numpy as np


def get_background_intensity(img_array, mask_array) -> np.array:
    """Get a stack of background intensities for every time step, using the top left well
    Returns a 1D numpy array of shape (num_steps,)
    """
    num_steps = img_array.shape[2]

    if not np.any(mask_array[0, 0]):
        backgrounds = np.mean(img_array[0, 0].reshape(num_steps, -1), axis=-1)
    else:
        # Fall back to median of entire image if there is not a blank in top left
        backgrounds = np.median(np.swapaxes(img_array, 2, 0).reshape(num_steps, -1), axis=-1)

    assert len(backgrounds.shape) == 1
    assert len(backgrounds) == img_array.shape[2]

    return backgrounds


def compute_all_y2_averaged(img_array, mask_array) -> np.array:
    """Compute average Y2 for each well in an entire plate
    Returns a 3D numpy array of shape (Ni, Nj, num_steps)
    """
    Ni, Nj = img_array.shape[:2]

    # First subtract background light intensity from each time point
    backgrounds = get_background_intensity(img_array, mask_array)
    img_array = img_array - backgrounds[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]

    # TODO: smooth each 2D image?

    # Compute pixelwise Y2 values, for every pixel, ignoring mask
    Fm_prime_array = img_array[:, :, 3::2, ...]  # Skip Fm
    F_array = img_array[:, :, 2::2, ...]  # Skip F0
    y2_array = (Fm_prime_array - F_array) / Fm_prime_array
    num_steps = Fm_prime_array.shape[2]
    assert num_steps == F_array.shape[2]

    # Set pixels outside mask to nan, and take mean of non-nan pixels
    mask_array = np.broadcast_to(mask_array[:, :, np.newaxis, ...], (Ni, Nj, num_steps, mask_array.shape[2], mask_array.shape[-1]))
    y2_array[~mask_array] = np.nan
    y2_array_means = np.nanmean(y2_array.reshape(Ni, Nj, num_steps, -1), axis=-1)

    assert y2_array_means.shape == (Ni, Nj, num_steps)
    assert np.nanmax(y2_array_means) < 1.0
    assert np.nanmin(y2_array_means) > -0.1, np.nanmin(y2_array_means)

    return y2_array_means
