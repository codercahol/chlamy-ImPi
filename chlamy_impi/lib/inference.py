import numpy as np


def compute_photosynthetic_params(cleaned_imgs):
    dark_imgs = cleaned_imgs[:, :, 0::2, :, :]
    light_imgs = cleaned_imgs[:, :, 1::2, :, :]
    # Re-arrange indices so time is in front
    Fs = np.moveaxis(dark_imgs, 2, 0)
    Fm_primes = np.moveaxis(light_imgs, 2, 0)

    # These arrays are all 0s
    # but we should do some bkgnd subtraction here
    # maybe subtract the noise thresholds?
    # blank_ctrl_dark = Fs[:,0,0,:,:]
    # blank_ctrl_light = Fm_primes[:,0,0,:,:]

    F0 = Fs[0, :, :, :, :]  # - blank_ctrl_dark[0]
    Fm = Fm_primes[0, :, :, :, :]  # - blank_ctrl_light[0]

    # compute pixel-wise Fv/Fm
    # TODO - suppress runtime warnings
    # (they are dealt with)
    Fv = Fm - F0
    QEY_pixel = Fv / Fm  # quantum yield of PSII
    QEY_pixel[np.isnan(QEY_pixel)] = 0
    # HACK - filter out 0.95
    # QEY_pixel[QEY_pixel < 0.95] = 0
    nonzero_pixels_QEY = np.sum(QEY_pixel > 0, axis=(2, 3))
    total_fluor_QEY = np.sum(QEY_pixel, axis=(2, 3))
    QEY = total_fluor_QEY / nonzero_pixels_QEY

    Fs = Fs[1:, :, :]  # - ctrl_Fs
    Fm_primes = Fm_primes[1:, :, :]  # - ctrl_Fm_prime

    # TODO - suppress runtime warnings
    # (caused by thresholding and the blank ctrl in top left)
    YII_pixel = (Fm_primes - Fs) / Fm_primes
    YII_pixel[np.isnan(YII_pixel)] = 0
    nonzero_pixels_YII = np.sum(YII_pixel > 0, axis=(3, 4))
    total_fluor_YII = np.sum(YII_pixel, axis=(3, 4))
    YII = total_fluor_YII / nonzero_pixels_YII

    # TODO: validate these
    NPQ = (Fm - Fm_primes) / Fm_primes
    Y_NPQ = Fs / Fm * NPQ

    return (QEY, YII, NPQ, Y_NPQ)
