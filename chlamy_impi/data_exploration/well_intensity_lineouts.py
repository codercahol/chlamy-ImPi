from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

INPUT_DIR = Path("./../../output/image_processing/v5/img_array")


def get_lineout(arr):
    max_point = np.unravel_index(np.argmax(arr), arr.shape)
    vals = arr[max_point[0], max_point[1] - 7: max_point[1] + 8]
    return vals


def plot_all_horizontal_lineouts(filename_to_all_vals, group):
    outdir = Path("./../../output/image_processing/lineouts")
    outdir.mkdir(exist_ok=True)

    num_plots = 6

    fig, axs = plt.subplots(1, num_plots, figsize=(24, 5), sharey=True)

    for i, (k, v) in enumerate(filename_to_all_vals.items()):
        try:
            ax = axs[i]
        except:
            ax = axs

        all_vals = v

        for vals in all_vals:
            ax.plot(vals, linewidth=3, color="black", alpha=0.1)

        ax.set_ylim(0, 400)
        ax.set_xlabel("Distance [pixels]")
        ax.set_title(k.stem)

    try:
        axs[0].set_ylabel("Fluorescence intensity [au]")
    except:
        axs.set_ylabel("Fluorescence intensity [au]")

    fig.tight_layout()
    fig.savefig(outdir / f"{group}.png")
    plt.close()


def main():
    assert INPUT_DIR.exists()
    filenames = list(INPUT_DIR.glob("*.npy"))

    def parse_name(f):
        parts = f.split(" ")
        parts = parts[1].split("-")

        plate_num = str(int(parts[0]))
        measurement_num = parts[1]

        return plate_num, measurement_num

    filenames.sort(key=lambda f: parse_name(str(f))[1])

    plate_groups = {parse_name(str(f)): f for f in filenames}

    group_to_filenames = defaultdict(list)

    for k, v in plate_groups.items():
        group = k[0]
        group_to_filenames[group].append(v)

    for group, filenames in group_to_filenames.items():
        filename_to_all_vals = defaultdict(list)

        for filename in filenames:
            img_array = np.load(filename)

            for i in range(img_array.shape[0]):
                for j in range(img_array.shape[1]):
                    arr = img_array[i, j, 0]
                    vals = get_lineout(arr)

                    filename_to_all_vals[filename].append(vals)

        plot_all_horizontal_lineouts(filename_to_all_vals, group)






if __name__ == "__main__":
    main()