from pathlib import Path
import logging

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from chlamy_impi.lib.fv_fm_functions import compute_all_fv_fm_pixelwise
from chlamy_impi.lib.mask_functions import compute_threshold_mask

logger = logging.getLogger(__name__)

INPUT_DIR = Path("./../../output/image_processing/v6/img_array")


def plot_all_fv_fm(filename, fv_fm_array):
    outdir = Path("./../../output/image_processing/fv_fm_distributions/v1")
    outdir.mkdir(exist_ok=True, parents=True)

    shape = fv_fm_array.shape[:2]

    assert shape == (16, 24)

    fig, axs = plt.subplots(shape[0], shape[1], figsize=(17, 12), dpi=250, sharex=True, sharey=True)

    assert fv_fm_array[0, 0].shape == (20, 20)
    assert np.nanmax(fv_fm_array) <= 1.0
    assert np.nanmin(fv_fm_array) >= 0.0

    for i in range(shape[0]):
        for j in range(shape[1]):
            ax = axs[i, j]
            ax.axis("off")
            im = ax.imshow(fv_fm_array[i, j], vmin=0.5, vmax=0.8, cmap="YlGnBu")
            #ax.text(0, 0, f"{np.nanmean(fv_fm_array[i, j]):.2f}", size="medium")

    fig.colorbar(im, ticks=[0.5, 0.8])

    fig.suptitle(filename.stem)
    fig.tight_layout()
    fig.savefig(outdir / f"{filename.stem}.png")
    plt.close()


def parse_name(f):
    parts = f.split(" ")
    parts = parts[1].split("-")

    plate_num = str(int(parts[0]))
    measurement_num = parts[1]

    return plate_num, measurement_num


def compute_fv_fm_stds(fv_fm_array):
    pixels_flat = fv_fm_array.reshape(fv_fm_array.shape[0], fv_fm_array.shape[1], -1)
    stds = np.nanstd(pixels_flat, axis=-1)
    return stds


def main():
    assert INPUT_DIR.exists()
    filenames = list(INPUT_DIR.glob("*.npy"))

    rows = []

    for filename in filenames:
        logger.info(f"filename: {filename}")
        img_array = np.load(filename)

        mask_array = compute_threshold_mask(img_array)
        fv_fm_array = compute_all_fv_fm_pixelwise(img_array, mask_array)

        fv_fm_std = np.nanmean(compute_fv_fm_stds(fv_fm_array))
        fv_fm_std_std = np.nanstd(compute_fv_fm_stds(fv_fm_array))
        logger.info(f"Average std is {fv_fm_std:.3f} +/- {fv_fm_std_std:.3f}")

        plot_all_fv_fm(filename, fv_fm_array)

        rows.append({"average std": fv_fm_std, "std of std": fv_fm_std_std})

    df = pd.DataFrame(rows)
    df.to_csv("fv_fm_spatial_stds.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
