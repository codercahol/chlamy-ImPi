import numpy as np
from skimage import filters


def compute_pixelwise_fv_fm(arr_0, arr_1, arr_mask, cntrl_f0, cntrl_fm) -> np.array:
    """Compute fv/fm value for each pixel given a single well.
    Returns a 1D array of fv/fm values for pixels inside the mask.
    """
    assert arr_mask.shape == arr_0.shape
    assert arr_mask.shape == arr_1.shape

    arr_0 = filters.gaussian(arr_0, sigma=1, channel_axis=None)
    arr_1 = filters.gaussian(arr_1, sigma=1, channel_axis=None)

    f0_arr = arr_0[arr_mask] - cntrl_f0
    fm_arr = arr_1[arr_mask] - cntrl_fm
    fv_arr = fm_arr - f0_arr

    return fv_arr / fm_arr


def compute_all_fv_fm_averaged(img_array, mask_array) -> np.array:
    """Compute average fv/fm for each well in an entire plate"""
    cntrl_f0, cntrl_fm = get_background_intensity(img_array, mask_array)

    all_fv_fm = np.zeros(shape=img_array.shape[:2], dtype=np.float32)

    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            all_fv_fm[i, j] = np.mean(
                compute_pixelwise_fv_fm(
                    img_array[i, j, 0],
                    img_array[i, j, 1],
                    mask_array[i, j],
                    cntrl_f0,
                    cntrl_fm,
                )
            )

    return all_fv_fm


def compute_all_fv_fm_pixelwise(img_array, mask_array) -> np.array:
    """Compute pixelwise fv/fm for each well in an entire plate. Outside the mask, the value is set to NaN."""
    cntrl_f0, cntrl_fm = get_background_intensity(img_array, mask_array)

    all_fv_fm = np.zeros_like(img_array[:, :, 0, ...])

    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            fv_fm_nonzero = compute_pixelwise_fv_fm(
                img_array[i, j, 0],
                img_array[i, j, 1],
                mask_array[i, j],
                cntrl_f0,
                cntrl_fm,
            )

            all_fv_fm[i, j][mask_array[i, j]] = fv_fm_nonzero
            all_fv_fm[i, j][~mask_array[i, j]] = np.nan

    return all_fv_fm


def get_background_intensity(img_array, mask_array):
    if not np.any(mask_array[0, 0]):
        cntrl_f0 = np.mean(img_array[0, 0, 0])  # Use mean of blank well
        cntrl_fm = np.mean(img_array[0, 0, 1])
    else:
        cntrl_f0 = np.median(
            img_array[:, :, 0, ...]
        )  # Fall back to global median intensity
        cntrl_fm = np.median(img_array[:, :, 1, ...])

    assert (
        abs(cntrl_f0 - cntrl_fm) < 20.0
    ), f"f0 control: {cntrl_f0}, fm control: {cntrl_fm}"

    return cntrl_f0, cntrl_fm
