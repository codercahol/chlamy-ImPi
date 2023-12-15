# %%
%load_ext autoreload
%autoreload 2

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from skimage import io
from lib import utils
from lib import data_processing as ip
from lib.constants4blank import *
import lib.inference as inf
from loguru import logger

# %% Load the Image

light_curve = "../data/20230920 light curve 384 wtf.tif"
blank_calib = "../data/blank dark light dark new calibration.tif"
WT_calib = "../data/WT dark light dark new calibration.tif"
# "../data/20230323 3h-3h Y(II).tif"

img_path = WT_calib
# 0-4 are static
# 5-19 are one plate
# 20-30 are another plate
# plate was taken out and put back in for each measurement

# assume for now that I always get 4 more images than explicitly planned on camera script:
# one pair is F0 and Fm
# the other pair are duds that immediately follow

tif = io.imread(img_path)  # open tiff file in read mode
# %% Filter out failed photos

tif, _ = ip.remove_failed_photos(tif)
# TODO - account for extra time due to missing images

# %% Apply a Gaussian filter to remove sensor noise
tif_blurred = ip.gaussian_blur(tif, kernel_size = 3, sigma = 1)

# %% Visualizing the data
# look at images from 3 randomly-selected times
ip.display_n(tif_blurred, 3)

# %% Hyperparameters of the approach

# TODO - fix
# should be the number of images divided by 2 bc we have F0 and Fm
NUM_TIMESTEPS = tif.shape[0] #int(tif.shape[0] / 2)
logger.debug("Hyperparameters\nNUM_TIMESTEPS: {}\nWidth (x,y): ({},{})".format(
    NUM_TIMESTEPS, CELL_WIDTH_X, CELL_WIDTH_Y
))

# upscale - HACK
pre_upscale = tif_blurred.unsqueeze(0)
pre_upscale_unblurred = torch.from_numpy(tif.astype(np.float32)).unsqueeze(0)
upscaled_unblurred = F.interpolate(pre_upscale_unblurred, scale_factor=2, mode='bilinear', align_corners=True)[0]
upscaled = F.interpolate(pre_upscale, scale_factor=2, mode='bilinear', align_corners=True)[0]
tif = upscaled_unblurred
tif_blurred = upscaled

# total HACK
# tif_blurred = tif_blurred - avg_img.unsqueeze(0)

# %% Visualize the crops we defined on a random image

# Pick a random image
# indexes (9, 75, 41) are good for testing noise thresholding
idx = np.random.randint(0, NUM_TIMESTEPS)
logger.debug("Randomly chosen index: {}".format(idx))
idx = 20
img = tif_blurred[idx,:,:]



croplines_img = ip.visualize_grid_crop(img)
save_image(
        torch.from_numpy(croplines_img).float().permute(2, 0, 1),
        "../output/intermediate/nb_gridlines.png",
    )
plt.imshow(croplines_img)

# %% Crop the images and the blurred images

#crops = ip.grid_crop(torch.from_numpy(tif.astype(np.float32)), NUM_TIMESTEPS)
constrain = False
# HACK-y
if constrain:
    start = 4
    endd = 20
    tif_past_t0 = tif[start:endd,:,:]
    tif_blurred_past_t0 = tif_blurred[start:endd,:,:]
    NUM_TIMESTEPS = endd - start
else:
    tif_past_t0 = tif.clone()
    tif_blurred_past_t0 = tif_blurred.clone()


crops = ip.grid_crop(tif_past_t0, NUM_TIMESTEPS)
crops_blurred = ip.grid_crop(tif_blurred_past_t0, NUM_TIMESTEPS)  

#  Estimate Noise using the blank control
plt.imshow(crops[1,0,1])
dark_idxs = range(0, NUM_TIMESTEPS, 2)
light_idxs = range(1, NUM_TIMESTEPS, 2)
# Not relevant for 384 plate of WTs bc there was no blank
mean_lit = crops[light_idxs,0,0].mean()
std_lit = crops[light_idxs,0,0].std()
mean_dark = crops[dark_idxs,0,0].mean()
std_dark = crops[dark_idxs,0,0].std()
mean = crops[:,0,0].mean()
std = crops[:,0,0].std()
THRESHOLD_lit = 20#40  # mean_lit + 3 * std_lit
THRESHOLD_dark = 16#26  # mean_dark + 3 * std_dark
logger.debug(f"LIGHT> mean : {mean_lit}, std {std_lit}, threshold {THRESHOLD_lit}")
logger.debug(f"DARK> mean : {mean_dark}, std {std_dark}, threshold {THRESHOLD_dark}")
# %% create masks per crop

# minimum value of a pixel across all timesteps must be > threshold
THRESHOLD_lit = 30
THRESHOLD_dark = 30
min_pixel = crops_blurred.min(dim=0)[0]
min_pixel_light = crops_blurred[light_idxs].min(dim=0)[0]
min_pixel_dark = crops_blurred[dark_idxs].min(dim=0)[0]
masks_light = min_pixel_light > THRESHOLD_lit
masks_dark = min_pixel_dark > THRESHOLD_dark
# manipulate shape of tensor to be (num_crops, 1, height, width)
# so that it can be saved as an image (idk why though)
mask_log_light = masks_light.float().view(-1,masks_light.shape[2], masks_light.shape[3]).unsqueeze(1)
mask_log_dark = masks_dark.float().view(-1,masks_dark.shape[2], masks_dark.shape[3]).unsqueeze(1)
save_image(mask_log_light, "../output/intermediate/nb_noise_mask_light.png", nrow=NUM_COLUMNS)
save_image(mask_log_dark, "../output/intermediate/nb_noise_mask_dark.png", nrow=NUM_COLUMNS)

np_img_light = plt.imread("../output/intermediate/nb_noise_mask_light.png")
np_img_dark = plt.imread("../output/intermediate/nb_noise_mask_dark.png")
plt.imshow(np_img_light)
plt.imshow(np_img_dark)

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
masks_light_shrunk = masks_light * sphere.unsqueeze(0).unsqueeze(0).unsqueeze(0)
masks_dark_shrunk = masks_dark * sphere.unsqueeze(0).unsqueeze(0).unsqueeze(0)
# again, reformat masks for saving as an image
mask_light_log = masks_light_shrunk.float().view(-1,masks_light.shape[2], masks_light.shape[3]).float().unsqueeze(1)
mask_dark_log = masks_dark_shrunk.float().view(-1,masks_dark.shape[2], masks_dark.shape[3]).float().unsqueeze(1)
save_image(mask_light_log, "../output/intermediate/masks_light_shrunk.png", nrow=NUM_COLUMNS)
save_image(mask_dark_log, "../output/intermediate/masks_dark_shrunk.png", nrow=NUM_COLUMNS)
# load and display the shrunk masks
np_img_light = plt.imread("../output/intermediate/masks_light_shrunk.png")
np_img_dark = plt.imread("../output/intermediate/masks_dark_shrunk.png")
plt.imshow(np_img_light)
plt.imshow(np_img_dark)

# %% Apply the masks to the crop

# apply the masks to the crops across each timestep
double_masks = torch.cat((masks_dark_shrunk, masks_light_shrunk), dim=0)
masks_shrunk = double_masks.repeat(int(NUM_TIMESTEPS/2), 1, 1, 1, 1)
crops_masked = crops_blurred * masks_shrunk


# %% Save processed data
# do Fv/Fm to a specific spot
cleaned_data = crops_masked.clone()
F0 = cleaned_data[0,:,:,:,:]
Fm = cleaned_data[1,:,:,:,:]
Fv = Fm - F0
QEY = Fv / Fm
QEY[torch.isnan(QEY)] = 0
# HACK - filter out 0.95
#QEY[QEY < 0.95] = 0
# 
sig_pix_cnt = torch.sum(QEY > 0, dim=(2,3))
flur =  torch.sum(QEY, dim = (2,3))
QEY_mean = flur / sig_pix_cnt
plt.imshow(QEY_mean, cmap="turbo"); plt.colorbar()

#crop_mean = torch.mean(crops_masked, dim=(3,4)) #inf.mean_fluorescence(crops_masked)
#utils.to_pickle(crop_mean, "../output/intensities_3h.pkl")
# %%
import seaborn as sns
#sns.set_theme()
sns.histplot(data=QEY_mean.view(-1), kde=True); plt.title("mean Fv/Fm for WTF")


# %% Check bias via QEY distribution across the plate
import lib.visualize as viz
(QEY, PY, NPQ, Y_NPQ) = inf.compute_photosynthetic_params(crop_mean)
viz.visualize_strain_param_across_plate(
    QEY, "Quantum Electron Yield", "../output/QEY.png"
)

# %% Note the misplating error rate
err = inf.misplating_error(crop_mean, num_blanks=7)
logger.debug("Misplating error rate: {}%".format(round(err.item() * 100, 2)))


# %% all together in one function

img_storage_folder = "../output/intermediate/"
pickle_file = "../output/2023-03-23_3hr_mean_fluor.pkl"
mf = ip.mean_fluorescences(
    ctrl_img_path, img_storage_folder,
    display=False,
    pickle_path=pickle_file
)