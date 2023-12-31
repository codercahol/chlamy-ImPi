from collections import defaultdict
from pathlib import Path
import logging

import numpy as np
from matplotlib import pyplot as plt

from chlamy_impi.lib.fv_fm_functions import compute_pixelwise_fv_fm
from chlamy_impi.lib.mask_functions import compute_threshold, compute_mask

logger = logging.getLogger(__name__)

INPUT_DIR = Path("./../../output/image_processing/v6/img_array")


def plot_all_fv_fm(filename_to_fv_fm, group):
    outdir = Path("./../../output/image_processing/fv_fm/v2")
    outdir.mkdir(exist_ok=True)

    fig, axs = plt.subplots(3, 1, figsize=(24, 10), dpi=250, sharex=1)

    xs = []
    ys = []
    cs = []
    x_to_vs = defaultdict(list)

    m_to_color = {"M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5, "M6": 6}

    for k, v in filename_to_fv_fm.items():
        filename = k[0]
        plate_num, measurement_num = parse_name(filename)
        i = k[1]
        j = k[2]

        xs.append(i * 24 + j)  # TODO: hardcoded number of cols here
        ys.append(v)
        cs.append(m_to_color[measurement_num[:2]])
        x_to_vs[i * 24 + j].append(v)

    x_to_stddev = {k: np.std(v) for k, v in x_to_vs.items()}
    x_to_m = {k: np.polyfit(range(len(v)), v, 1)[0] for k, v in x_to_vs.items()}

    ax = axs[0]
    scatter = ax.scatter(xs, ys, c=cs, label=cs, s=18)
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower right", title="Measurement number")
    ax.add_artist(legend1)
    ax.set_title(f"Plate {group}")
    ax.set_ylabel("Fv/Fm")
    ax.set_ylim(0, 0.8)

    ax = axs[1]
    ax.bar(x_to_stddev.keys(), x_to_stddev.values(), facecolor="orange")
    ax.set_ylabel("Stdev of Fv/Fm per sample")

    ax = axs[2]
    ax.bar(x_to_m.keys(), x_to_m.values(), facecolor="blue")
    ax.set_xlabel("Sample #")
    ax.set_ylabel("Trend in Fv/Fm across measurements")

    fig.tight_layout()
    fig.savefig(outdir / f"{group}.png")
    plt.close()


def parse_name(f):
    parts = f.split(" ")
    parts = parts[1].split("-")

    plate_num = str(int(parts[0]))
    measurement_num = parts[1]

    return plate_num, measurement_num


def main():
    assert INPUT_DIR.exists()
    filenames = list(INPUT_DIR.glob("*.npy"))


    # This orders the filenames according to M1, M2, etc.
    # As a result, when we come to do the plotting, the plots are also ordered in this way
    # Note: this relies on python dictionaries being ordered according to key insertion order, which afaik is not
    # part of the standard, but happens to be implemented in cpython.
    filenames.sort(key=lambda f: parse_name(str(f))[1])

    plate_groups = {parse_name(str(f)): f for f in filenames}

    group_to_filenames = defaultdict(list)

    for k, v in plate_groups.items():
        group = k[0]
        group_to_filenames[group].append(v)

    for group, filenames in group_to_filenames.items():
        filename_to_fv_fm = {}

        logger.info(f"Looking at group {group}")

        for filename in filenames:
            logger.info(f"filename: {filename}")
            img_array = np.load(filename)

            try:
                threshold = compute_threshold(img_array[0, 0])
            except ValueError:
                logger.warning(f"File: {filename}. No blank well found, skipping")
                continue

            logger.debug(f"Computed threshold for plate as: {threshold}")

            # Extract control images by smoothing blamk well
            #cntrl_0 = filters.gaussian(img_array[0, 0, 0], sigma=1, channel_axis=None)
            #cntrl_1 = filters.gaussian(img_array[0, 0, 1], sigma=1, channel_axis=None)

            cntrl_0 = np.mean(img_array[0, 0, 0])  # Better to use mean of blank well to estimate signal
            cntrl_1 = np.mean(img_array[0, 0, 1])
            assert abs(cntrl_0 - cntrl_1) < 10.0, f"0: {cntrl_0}, 1: {cntrl_1}"

            all_fv_fm = []
            all_fv_fm_std = []

            for i in range(img_array.shape[0]):
                for j in range(img_array.shape[1]):
                    arr_0 = img_array[i, j, 0]  # Corresponds to F0
                    arr_1 = img_array[i, j, 1]  # Corresponds to Fm
                    arr_mask = compute_mask(img_array[i, j], threshold)

                    if np.any(arr_mask):
                        fv_fm_arr = compute_pixelwise_fv_fm(arr_0, arr_1, arr_mask, cntrl_0, cntrl_1)
                        fv_fm_std = np.std(fv_fm_arr)
                        all_fv_fm_std.append(fv_fm_std)
                    else:
                        # Empty mask - set to 0
                        fv_fm_arr = np.zeros_like(arr_0)

                    # Compute a pixel-wise mean here
                    fv_fm = np.mean(fv_fm_arr)
                    all_fv_fm.append(fv_fm)

                    filename_to_fv_fm[(str(filename), i, j)] = fv_fm

            logger.info(f"Mean fv/fm = {np.mean(all_fv_fm)}")
            logger.info(f"Mean stdev of pixel fv/fm = {np.mean(all_fv_fm_std)}")

            assert np.mean(all_fv_fm_std) < 0.035

        plot_all_fv_fm(filename_to_fv_fm, group)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
