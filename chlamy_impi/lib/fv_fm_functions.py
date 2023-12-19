import numpy as np
from skimage import filters


def compute_pixelwise_fv_fm(arr_0, arr_1, arr_mask, cntrl_0, cntrl_1) -> np.array:
    """Compute fv/fm value for each pixel given a single well
    """
    assert arr_0.shape == (20, 20)
    assert arr_mask.shape == arr_0.shape
    assert arr_mask.shape == arr_1.shape
    #assert arr_mask.sum() >= 4

    arr_0 = filters.gaussian(arr_0, sigma=1, channel_axis=None)
    arr_1 = filters.gaussian(arr_1, sigma=1, channel_axis=None)

    f0_arr = arr_0[arr_mask] - cntrl_0
    fm_arr = arr_1[arr_mask] - cntrl_1
    fv_arr = fm_arr - f0_arr

    return fv_arr / fm_arr


def compute_all_fv_fm(img_array, mask_array) -> np.array:
    """Compute average fv/fm for each well in an entire plate
    """
    cntrl_0, cntrl_1 = get_background_intensity(img_array, mask_array)

    all_fv_fm = np.zeros(shape=img_array.shape[:2], dtype=np.float32)

    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            all_fv_fm[i, j] = np.mean(compute_pixelwise_fv_fm(
                img_array[i, j, 0],
                img_array[i, j, 1],
                mask_array[i, j],
                cntrl_0,
                cntrl_1
            ))

    return all_fv_fm


def get_background_intensity(img_array, mask_array):
    if not np.any(mask_array[0, 0]):
        cntrl_0 = np.mean(img_array[0, 0, 0])  # Use mean of blank well
        cntrl_1 = np.mean(img_array[0, 0, 1])
    else:
        cntrl_0 = np.median(img_array[:, :, 0, ...])  # Fall back to global median intensity
        cntrl_1 = np.median(img_array[:, :, 1, ...])

    assert abs(cntrl_0 - cntrl_1) < 20.0, f"0: {cntrl_0}, 1: {cntrl_1}"

    return cntrl_0, cntrl_1
