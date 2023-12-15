# adapted from
# https://github.com/ThibaultGROUEIX/Fluorescence/blob/main/explore_data.ipynb
# exploring the code

# %%
%load_ext autoreload
%autoreload 2

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import torch
from torchvision.utils import save_image
from skimage import io
from lib import utils
from lib import data_processing as ip
from lib.constants import config
import lib.inference as inf
from loguru import logger

# %% Load the Image

tiff_file = "../data/20230323 3h-3h Y(II).tif"
tif = io.imread(tiff_file)  # open tiff file in read mode
# %% Load Constants

(
    NUM_ROWS,
    NUM_COLUMNS,
    X_MIN,
    CELL_WIDTH_X,
    X_MAX,
    Y_MIN,
    CELL_WIDTH_Y,
    Y_MAX,
) = config("test_plate")

# %% Filter out failed photos

tif, _ = ip.remove_failed_photos(tif)
# TODO - account for extra time due to missing images

# %% Apply a Gaussian filter to remove sensor noise
tif_blurred = ip.gaussian_blur(tif, kernel_size = 3, sigma = 1)

# %% Visualizing the data
# look at images from 3 randomly-selected times
ip.display_n(tif_blurred, 3)

# %% Hyperparameters of the approach

# TODO - fix - TODO
# should be the number of images divided by 2::Int bc we have F0 and Fm
NUM_TIMESTEPS = tif.shape[0]
logger.debug("Hyperparameters\nNUM_TIMESTEPS: {}\nWidth (x,y): ({},{})".format(
    NUM_TIMESTEPS, CELL_WIDTH_X, CELL_WIDTH_Y
))

# %% Visualize the crops we defined on a random image

# Pick a random image
# indexes (9, 75, 41) are good for testing noise thresholding
idx = np.random.randint(0, NUM_TIMESTEPS)
logger.debug("Randomly chosen index: {}".format(idx))
img = tif_blurred[idx,:,:]


croplines_img = ip.visualize_grid_crop(img)
save_image(
        torch.from_numpy(croplines_img).float().permute(2, 0, 1),
        "../output/intermediate/nb_gridlines.png",
    )
plt.imshow(croplines_img)

# %% Crop the images and the blurred images

crops = ip.grid_crop(torch.from_numpy(tif.astype(np.float32)), NUM_TIMESTEPS)
crops_blurred = ip.grid_crop(tif_blurred, NUM_TIMESTEPS)  

#  Estimate Noise using the blank control
plt.imshow(crops[0,0,0])
mean = crops[:,0,0].mean()
std = crops[:,0,0].std()
THRESHOLD = mean + 3 * std
logger.debug(f"mean : {mean}, std {std}, threshold {THRESHOLD}")
# %% create masks per crop

# minimum value of a pixel across all timesteps must be > threshold
min_pixel = crops_blurred.min(0)[0]
masks = min_pixel > THRESHOLD
# manipulate shape of tensor to be (num_crops, 1, height, width)
# so that it can be saved as an image (idk why though)
mask_log = masks.float().view(-1,masks.shape[2], masks.shape[3]).unsqueeze(1)
save_image(mask_log, "../output/intermediate/nb_noise_mask.png", nrow=NUM_COLUMNS)

np_img = plt.imread("../output/intermediate/nb_noise_mask.png")
plt.imshow(np_img)

# %% Shrink masks to avoid contagion effects

# Create a disk-like mask
sphere = ip.disk_mask(CELL_WIDTH_X, CELL_WIDTH_Y)
plt.imshow(sphere)
# the x/y axes are deliberately flipped here
plt.ylim(0, CELL_WIDTH_X)
plt.xlim(0, CELL_WIDTH_Y)

# %%
# Compute the intersection of the masks with the disk
# add dimensions to the sphere mask so it can be applied to each crop
masks_shrunk = masks * sphere.unsqueeze(0).unsqueeze(0).unsqueeze(0)
# again, reformat masks for saving as an image
mask_log = masks_shrunk.float().view(-1,masks.shape[2], masks.shape[3]).float().unsqueeze(1)
save_image(mask_log, "../output/intermediate/masks_shrunk.png", nrow=NUM_COLUMNS)
# load and display the shrunk masks
np_img = plt.imread("../output/intermediate/masks_shrunk.png")
plt.imshow(np_img)

# %% Apply the masks to the crop

# apply the masks to the crops across each timestep
crops_masked = crops_blurred * masks_shrunk.unsqueeze(0)
crops_masked = crops_masked[0,:,:,:,:,:]


# %% Save processed data
crop_mean = torch.mean(crops_masked, dim=[3,4])
utils.to_pickle(crop_mean, "../output/intensities_3h.pkl")

# %% Note the misplating error rate
err = inf.misplating_error(crop_mean, num_blanks=7)
logger.debug("Misplating error rate: {}%".format(round(err.item() * 100, 2)))


# %% all together in one function

img_storage_folder = "../output/intermediate/"
pickle_file = "../output/2023-03-23_3hr_mean_fluor.pkl"
mf = ip.mean_fluorescences(
    tiff_file, img_storage_folder,
    display=False,
    pickle_path=pickle_file
)
