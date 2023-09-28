import torch
from lib import utils
from lib.constants import NUM_ROWS, NUM_COLUMNS
import numpy as np


def mean_fluorescence(crops):
    """
    Compute the mean fluorescence across non-masked pixels in each crop.
    Input:
        crops: 3D numpy array of shape
                (num_timesteps, num_rows, num_columns, X_num_pixels, Y_num_pixels)
    Output:
        mean_fluorescences: 3D numpy array of shape (num_timesteps, num_rows, num_columns)
    """
    signal_pixel_counts = torch.sum(crops > 0, dim=(3, 4))
    fluorescences = torch.sum(crops, dim=(3, 4))
    mean_fluorescences = fluorescences / signal_pixel_counts
    return mean_fluorescences


# TODO - undo HACK: removal of ctrl in top left corner
def compute_photosynthetic_params(mean_fluorescences):
    if type(mean_fluorescences) == str:
        # or
        # mf = utils.from_pickle("../output/intensities_3h.pkl")
        mean_fluorescences = torch.load(mean_fluorescences)
    mf = mean_fluorescences
    ctrl = mf[:, 0, 0]
    ctrl_Fs = ctrl[2:102:2, np.newaxis, np.newaxis]
    ctrl_Fm_prime = ctrl[3:102:2, np.newaxis, np.newaxis]

    F0 = mf[0, :, :]  # - ctrl[0]
    Fm = mf[1, :, :]  # - ctrl[1]
    Fv = Fm - F0
    QEY = Fv / Fm  # quantum yield of PSII
    Fs = mf[2:102:2, :, :]  # - ctrl_Fs
    Fm_prime = mf[3:102:2, :, :]  # - ctrl_Fm_prime
    PY = (Fm_prime - Fs) / Fm_prime
    NPQ = (Fm - Fm_prime) / Fm_prime
    Y_NPQ = Fs / Fm * NPQ
    return (QEY, PY, NPQ, Y_NPQ)


def misplating_error(data, num_blanks):
    """
    Estimate the error due to misplating, which results in wells with no growing cells.
    We assume that wells with a noise-corrected mean fluorescence == 0 are misplated.
    Input:
        data: 3D numpy array of shape (num_timesteps, num_rows, num_columns)
        num_blanks: number of blanks in the plate
    """
    total_wells = NUM_ROWS * NUM_COLUMNS - num_blanks
    well_means = torch.mean(data, dim=[0])
    num_misplated = torch.sum(well_means == 0) - num_blanks
    misplated_error = num_misplated / total_wells
    return misplated_error
