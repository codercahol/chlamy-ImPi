# %%
%load_ext autoreload
%autoreload 2

from pathlib import Path

from segment_multiwell_plate import segment_multiwell_plate
# from tqdm import tqdm
import pandas as pd
import platform

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline
from lib import utils
from lib import data_processing as ip
import lib.inference as inf
from loguru import logger
# %% Load Constants

# Fix for matplotlib memory leak, see https://github.com/matplotlib/matplotlib/issues/20067
if platform.system() == "Linux":
    logger.info("Setting matplotlib backend to agg")
    matplotlib.use("agg")

# For now, input and output dir hard-coded here
INPUT_DIR = Path("../data")
OUTPUT_DIR = Path("../output/image_processing/v3")
USE_MULTIPROCESSING = False
OUTPUT_VISUALISATIONS = True

# %% Load the Image

img_path = "../data/20231213 9-M5_2h-2h.tif"


# %% new fangled way

# %% find all images
logger.info("\n" + "=" * 32 + "\nStarting image_processing.py...\n" + "=" * 32)
utils.validate_inputs(INPUT_DIR)
filenames = utils.find_all_images(INPUT_DIR)

sep = "\n\t"
logger.info(
    f"Found a total of {len(filenames)} tif files: \n\t{sep.join(str(x) for x in filenames)}"
)

# %% process each image

# %% initial screening
failed_files = []
# dealing with only 1 for simplicity
filename = filenames[1]

name = filename.stem

if utils.results_dir_path(name, OUTPUT_DIR).exists():
    logger.info(f"Image {name} has already been processed. Skipping.")
    # continue
else:
    utils.results_dir_path(name, OUTPUT_DIR).mkdir(parents=True)

# %%

# try:
#except AssertionError as e:
#    logger.error(f"File: {filename.stem}. Error: {e}")
#    failed_files.append((filename.stem, e))

# logger.info(f"Failed on {len(failed_files)} tif files: {failed_files}")
# logger.info("Program completed normally")

# %% load and remove blanks
tif = utils.load_image(filename)

logger.debug(f"NUM_TIMESTEPS={tif.shape[0]}")

tif, num_blank_frames = ip.remove_failed_photos(tif)

# %% segment the plate

img_array, well_coords, i_vals, j_vals = segment_multiwell_plate(
    tif, peak_finder_kwargs={"peak_prominence": 0.1}, output_full=True
)

# %% apply masks 

dark_threshold = ip.estimate_noise_threshold(img_array, lighting="dark")
light_threshold = ip.estimate_noise_threshold(img_array, lighting="light")
logger.info(f"Dark threshold = {dark_threshold}")
logger.info(f"Light threshold = {light_threshold}")

disk_mask = ip.disk_mask(img_array)

NUM_SAMPLES = img_array.shape[2]
NUM_TIMESTEPS = NUM_SAMPLES / 2
dark_idxs = range(0, NUM_SAMPLES, 2)
light_idxs = range(1, NUM_SAMPLES, 2)
dark_imgs = img_array[:, :, dark_idxs, :, :]
light_imgs = img_array[:, :, light_idxs, :, :]

dark_mask = ip.generate_mask(
    dark_imgs,
    dark_threshold,
    disk_mask,
)
light_mask = ip.generate_mask(
    light_imgs,
    light_threshold,
    disk_mask,
)

total_mask = dark_mask & light_mask
masked_imgs = img_array * total_mask


# %% count empty wells & save data

empty_wells, total_wells = ip.count_empty_wells(total_mask)

logger.info(f"Found a total of {empty_wells} / {total_wells} empty wells")

num_overlapping = ip.validate_well_mask_array(total_mask)

# save masked imgs
np.save(utils.results_dir_path(name, OUTPUT_DIR) / "masked_imgs.npy", masked_imgs)

plate_info = {
    "name": name,
    "timepoints": NUM_TIMESTEPS,
    "num_blank_frames": num_blank_frames,
    "dark_threshold": dark_threshold,
    "light_threshold": light_threshold,
    "total_wells": total_wells,
    "num_empty_wells": empty_wells,
    "num_overlapping_masks": num_overlapping,
}

df = utils.load_plate_info(OUTPUT_DIR)
new_row = pd.DataFrame([plate_info])
df = pd.concat([df, new_row], ignore_index=True)
utils.write_plate_info(df, OUTPUT_DIR)

# %% compute photosynthetic params

(QEY, YII, NPQ, Y_NPQ) = inf.compute_photosynthetic_params(masked_imgs)

# %% display some visuals 

if OUTPUT_VISUALISATIONS:
    ip.visualise_channels(tif, savedir=utils.results_dir_path(name, OUTPUT_DIR) / "raw")
    ip.visualise_mask_array(
        masked_imgs, savedir=utils.results_dir_path(name, OUTPUT_DIR) / "masks"
    )
    ip.visualise_grid_crop(
        tif,
        img_array,
        i_vals,
        j_vals,
        well_coords,
        savedir=utils.results_dir_path(name, OUTPUT_DIR) / "grid",
    )