import numpy as np
import torch
import torchvision
from loguru import logger
import matplotlib.pyplot as plt
import cv2
from skimage import io
import torch.nn.functional as F

from lib import utils
from torchvision.utils import save_image
import openpyxl as xl
import pandas as pd

# TODO - this file is getting too long
# -- break up into multiple files?


def remove_failed_photos(tif):
    """
    Remove photos that are all black

    Input:
        tif: torch tensor of shape (num_images, height, width)
    Output:
        tif: torch tensor of shape (num_images, height, width)
        photo_index: numpy array of indices of photos that were kept
    """
    max_per_timestep = tif.max(1).max(1)
    keep_image = max_per_timestep > 0
    logger.info(
        f"Discarding {sum(~keep_image)} images (indices {list(np.argwhere(~keep_image))}) which are all black"
    )
    photo_index = np.arange(len(keep_image))[keep_image]
    tif = tif[keep_image]
    return tif, photo_index


def gaussian_blur(tif, kernel_size, sigma):
    """
    Apply a Gaussian kernel to blur noise in an image

    Input:
        tif: torch tensor of shape (num_images, height, width)
        kernel_size: the side length in pixels of the blurring kernel
        sigma: the standard deviation of the Gaussian kernel
    Output:
        tif_blurred: torch tensor of shape (num_images, height, width)
    """
    blurrer = torchvision.transforms.GaussianBlur(
        kernel_size=kernel_size, sigma=(sigma)
    )
    tif_blurred = blurrer(torch.from_numpy(tif.astype(np.float32)))
    return tif_blurred


def display_n(tif, n):
    """
    Display n random images from the tiff file

    Input:
        tif: torch tensor of shape (num_images, height, width)
        n: number of images to display
    Output:
        None (displays n images)
    """
    logger.debug("Shape {}".format(tif.shape))
    # pick n random images
    idx = np.random.randint(0, tif.shape[0], n)
    logger.debug("Randomly chosen indices: {}".format(idx))
    for i in idx:
        plt.imshow(tif[i, :, :])
        plt.show()


def display_n_crops(crops, n):
    """
    Display n random crops

    Input:
        crops: torch tensor of shape
            (num_timesteps, num_rows, num_columns, height, width)
        n: number of images to display
    Output:
        None (displays n images)
    """
    logger.debug("Shape {}".format(crops.shape))
    # pick n random images
    t_idx = np.random.randint(0, crops.shape[0], n)
    row_idx = np.random.randint(0, crops.shape[1], n)
    col_idx = np.random.randint(0, crops.shape[2], n)
    idxs = list(zip(t_idx, row_idx, col_idx))
    logger.debug("Randomly chosen indices (time, row, col): {}".format(idxs))
    for t, r, c in idxs:
        plt.imshow(crops[t, r, c, :, :])
        plt.show()


def draw_mark(cv_src, pts, pts2):
    """
    Draws rectangles on an image
    Input:
        cv_src: the input image
        pts: the top left corner of the rectangle
        pts2: the bottom right corner of the rectangle
        color: the color of the rectangle
    Output:
        None (modifies the input image)
    """
    _GOLD = (255, 127, 127)
    cv2.rectangle(cv_src, pts, pts2, color=_GOLD, thickness=1)


def generate_grid_crop_coordinates(const):
    """
    Generate the coordinates for the grid crop
    """
    # args are (start, end, num_points)
    # +1 is added to the number of points to include the end point
    # (eg. we want the number of fenceposts, not the number of fences)
    x = torch.linspace(const.X_MIN, const.X_MAX, const.NUM_ROWS + 1)
    y = torch.linspace(const.Y_MIN, const.Y_MAX, const.NUM_COLUMNS + 1)
    a, b = torch.meshgrid(x, y)
    a, b = a.int(), b.int()
    return a, b


def visualize_grid_crop(*args, **kwargs):
    # HACK - this should be removed once the two methods are consolidated
    if len(args) == 1:
        old_visualize_grid_crop(*args, **kwargs)
    elif len(args) == 6:
        new_visualise_grid_crop(*args, **kwargs)
    else:
        return TypeError("Number of arguments didn't match either method")


def old_visualize_grid_crop(img, threshold=35):
    """
    Visualize the grid crop
    """
    # threshold the image to set a limit on noise
    # and re-scale to full intensity range of 0-255
    work_img = 255 * (img.clone() > threshold)
    # convert to 3-channel image
    # (needed for working with torch.save_image which assumes...
    # an image with dimensions (channels, height, width))
    work_img = work_img.unsqueeze(0).repeat(3, 1, 1)
    # permute the dimensions for working with cv2 drawing functions
    input_img = work_img.permute(1, 2, 0).contiguous().numpy().astype(np.int16)
    a, b = generate_grid_crop_coordinates()
    for i in range(a.shape[0] - 1):
        for j in range(a.shape[1] - 1):
            pts = (b[i, j].item(), a[i, j].item())
            pts2 = (b[i + 1, j + 1].item(), a[i + 1, j + 1].item())
            draw_mark(input_img, pts, pts2, color=_GREEN)
    return input_img


def new_visualise_grid_crop(
    tif, img_array, i_vals, j_vals, well_coords, savedir, max_channels=5
):
    # This function needs more comments
    """
    Input:
        tif: torch tensor of shape (num_images, height, width)
        img_array: torch tensor of shape
            (num_timesteps, num_rows, num_columns, height, width)
        i_vals: the row indices of the grid crop?
        j_vals: the column indices of the grid crop?
        well_coords: the coordinates of the wells?
        savedir: the directory to save the plots in
        max_channels: the maximum number of channels to plot
    Output:
        None (saves plots to savedir)
    """
    logger.debug(f"Saving plots of grid crop in {savedir}")
    savedir.mkdir(parents=True, exist_ok=True)

    img_shape = tif.shape
    array_shape = img_array.shape

    iv, jv = np.meshgrid(i_vals, j_vals, indexing="ij")
    iv2, jv2 = np.meshgrid(i_vals, j_vals, indexing="xy")

    for channel in range(min(img_shape[0], max_channels)):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(tif[channel, :, :])
        # Draw well centre coords
        ax.scatter(
            list(zip(*well_coords))[1],
            list(zip(*well_coords))[0],
            color="red",
            marker="x",
            s=2,
        )
        # Draw grid
        ax.plot(jv, iv, color="red")
        ax.plot(jv2, iv2, color="red")
        fig.savefig(savedir / f"{channel}_grid.png")
        fig.clf()
        plt.close(fig)

        fig, axs = plt.subplots(array_shape[0], array_shape[1])
        for i, j in itertools.product(range(array_shape[0]), range(array_shape[1])):
            ax = axs[i, j]
            ax.axis("off")
            ax.imshow(
                img_array[i, j, channel],
                vmin=tif[channel].min(),
                vmax=tif[channel].max(),
            )
        fig.savefig(savedir / f"{channel}_subimage_array.png")
        fig.clf()
        plt.close(fig)


def visualise_channels(tif, savedir, max_channels=None):
    logger.debug(f"Writing out plots of all time points in {savedir}")

    savedir.mkdir(parents=True, exist_ok=True)

    shape = tif.shape

    if max_channels is None:
        max_channels = shape[0]

    for channel in range(min(shape[0], max_channels)):
        fig, ax = plt.subplots(1, 1)
        ax.imshow(tif[channel, :, :])
        fig.savefig(savedir / f"{channel}.png")
        fig.clf()
        plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(np.mean(tif, axis=0))
    fig.savefig(savedir / f"avg.png")
    fig.clf()
    plt.close(fig)


# TODO - check it works with the way I've been handling files
def validate_well_mask_array(mask_array) -> int:
    """Perform some checks on the well masks.

    Returns the number of well masks which overlap with the boundary of the sub-image of the well.
    """
    array_shape = mask_array.shape
    arr = np.zeros_like(mask_array[:, :, 0, 0])

    for i, j in itertools.product(range(array_shape[0]), range(array_shape[1])):
        if has_true_on_boundary(mask_array[i, j]):
            logger.warning(f"Mask {i},{j} has hit boundary")
            arr[i, j] = True

    num_overlapping = np.sum(arr)

    # assert num_overlapping <= 3, f"We have found overlapping masks for {num_overlapping} masks"
    logger.info(f"We have found overlapping masks for {num_overlapping} masks")

    return num_overlapping


# TODO - check it works with the way I've been handling files
def has_true_on_boundary(arr):
    """Check if mask reaches edge of cell - should always be false"""

    # Check the top and bottom rows
    if np.any(arr[0, :]) or np.any(arr[-1, :]):
        return True

    # Check the left and right columns
    if np.any(arr[:, 0]) or np.any(arr[:, -1]):
        return True

    return False


def grid_crop(const, img, num_timesteps):
    x = torch.round(torch.linspace(const.X_MIN, const.X_MAX, const.NUM_ROWS + 1)).to(
        torch.int16
    )
    y = torch.round(torch.linspace(const.Y_MIN, const.Y_MAX, const.NUM_COLUMNS + 1)).to(
        torch.int16
    )
    crops = torch.Tensor(
        size=(
            num_timesteps,
            const.NUM_ROWS,
            const.NUM_COLUMNS,
            const.CELL_WIDTH_X,
            const.CELL_WIDTH_Y,
        )
    )
    for i in range(len(x) - 1):
        for j in range(len(y) - 1):
            crops[:, i, j] = img[
                :,
                x[i] : x[i + 1],
                y[j] : y[j + 1],
            ].contiguous()
    # this might be redundant, but it was in the original code
    crops = crops.contiguous()
    return crops


def disk_mask(const, radius_fraction=4 / 5):
    """
    Create a disk-like mask
    """
    sphere = torch.zeros(size=(const.CELL_WIDTH_X, const.CELL_WIDTH_Y))
    smaller_diameter = min(const.CELL_WIDTH_X, const.CELL_WIDTH_Y)
    for i in range(sphere.shape[0]):
        for j in range(sphere.shape[1]):
            # formula for a circle centered at the center of the rectangle
            # with the given dimensions
            if (i - const.CELL_WIDTH_X / 2) ** 2 + (j - const.CELL_WIDTH_Y / 2) ** 2 < (
                radius_fraction * smaller_diameter / 2
            ) ** 2:
                sphere[i, j] = 1
    return sphere


# TODO - keep this or disk_mask?
def get_disk_mask(img_array):
    disk_radius = min(img_array.shape[-2:]) // 2
    disk_mask = morphology.disk(radius=disk_radius, dtype=bool)
    if disk_mask.shape[0] > img_array.shape[3]:
        assert (
            img_array.shape[3] == img_array.shape[4]
        )  # Check assumption of square cells
        disk_mask = disk_mask[:-1, :-1]
    assert disk_mask.shape == img_array.shape[-2:]
    return disk_mask


# TODO - integrate
def visualise_mask_array(mask_array, savedir):
    logger.debug(f"Writing out plot of masks to {savedir}")
    savedir.mkdir(parents=True, exist_ok=True)

    array_shape = mask_array.shape

    fig, axs = plt.subplots(array_shape[0], array_shape[1])
    for i, j in itertools.product(range(array_shape[0]), range(array_shape[1])):
        ax = axs[i, j]
        ax.axis("off")
        ax.imshow(mask_array[i, j])
    fig.savefig(savedir / "mask_array.png")
    fig.clf()
    plt.close(fig)


# TODO - integrate
def count_empty_wells(mask_array):
    """
    Estimate the error due to misplating, which results in wells with no growing cells.

    Input:
        mask_array: 4D numpy array of shape (num_rows, num_columns, height, width)
        num_blanks: number of blanks in the plate
    """
    mask_array_flat_im = mask_array.reshape(mask_array.shape[:2] + (-1,))
    total_wells = mask_array.shape[0] * mask_array.shape[1]
    num_good_wells = np.sum(np.max(mask_array_flat_im, axis=-1))
    empty_wells = total_wells - num_good_wells
    return empty_wells, total_wells


# TODO - integrate
def save_mean_array(mean_fluor_array, name):
    outfile = OUTPUT_DIR / "mean_arrays" / f"{name}.npy"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    np.save(outfile, mean_fluor_array)
    logger.info(f"Mean fluorescence array saved out to: {outfile}")

    # Also save as csv to be read using pandas
    rows = []
    for i, j in itertools.product(
        range(mean_fluor_array.shape[0]), range(mean_fluor_array.shape[1])
    ):
        col_to_val = {
            f"mean_fluorescence_frame_{k}": mean_fluor_array[i, j, k]
            for k in range(mean_fluor_array.shape[2])
        }
        col_to_val.update({"row": i, "col": j})
        rows.append(col_to_val)
    df = pd.DataFrame(rows)
    outfile = OUTPUT_DIR / "mean_arrays" / f"{name}.csv"
    df.to_csv(outfile)
    logger.info(f"Mean fluorescence array saved out to: {outfile}")


def compute_noise_threshold(ctrl_imgs, subset_idxs=None, display=False, n_stdDevs=5):
    if subset_idxs is not None:
        ctrl_imgs = ctrl_imgs[subset_idxs]
    if display:
        plt.imshow(ctrl_imgs[0])
    mean = ctrl_imgs.mean()
    std = ctrl_imgs.std()
    THRESHOLD = mean + n_stdDevs * std
    logger.debug(f"Noise threshold mean : {mean}, std: {std}. Threshold: {THRESHOLD}")
    return THRESHOLD


def log_hyperparameters(const, num_timesteps):
    logger.debug(
        "Hyperparameters\nNUM_TIMESTEPS: {}\nWidth (x,y): ({},{})".format(
            num_timesteps, const.CELL_WIDTH_X, const.CELL_WIDTH_Y
        )
    )


def normalize(img):
    """
    Normalize an image to the range [0, 1]
    """
    return (img - img.min()) / (img.max() - img.min())


def convert_to_3channel_img(twoD_img):
    """
    Convert a 2D image (height, width) to a 3D image (channels, H, W)
    with 3 channels (RGB)
    """
    threeD_img = twoD_img.unsqueeze(0).repeat(3, 1, 1)
    return threeD_img


def clean_mask_n_crop(
    const, tif_path, img_storage_path="", display=False, pickle_path=""
):
    if img_storage_path == "" and display:
        raise ValueError(
            "Images must be saved in order to be displayed. "
            + "Please enter a path for the images to be saved."
        )
    # logging will either (save) or (save & display) images
    should_log = img_storage_path != "" or display
    tif = io.imread(tif_path)
    tif, _ = remove_failed_photos(tif)
    tif_blurred = gaussian_blur(tif, 3, 1)
    if display:
        display_n(tif_blurred, 3)
    num_timesteps = tif.shape[0]
    log_hyperparameters(num_timesteps)
    crops = grid_crop(tif_blurred, num_timesteps)
    ctrl_imgs = crops[:, 0, 0]
    threshold = compute_noise_threshold(ctrl_imgs, display)
    # Checks that the crop and noise threshold look good when combined
    if should_log:
        # Pick a random image
        idx = np.random.randint(0, num_timesteps)
        logger.debug("Randomly chosen index: {}".format(idx))
        img = tif_blurred[idx, :, :]
        croplines_img = visualize_grid_crop(img, threshold)
        # re-format the image so that it has the corrects dim'ns
        # (channels, height, width); for being saved
        croplines_img_toSave = torch.from_numpy(
            croplines_img.astype(np.float32)
        ).permute(2, 0, 1)
        save_image(
            convert_to_3channel_img(normalize(img)), img_storage_path + "input_img.png"
        )
        save_image(croplines_img_toSave, img_storage_path + "gridlines_n_threshold.png")
        if display:
            plt.imshow(img)
            plt.imshow(croplines_img)
    # uses the same threshold as for the logged images
    noise_mask = crops.min(0)[0] > threshold
    shape_mask = disk_mask(const.CELL_WIDTH_X, const.CELL_WIDTH_Y)
    mask = noise_mask * shape_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    if should_log:
        # manipulate shape of tensor to be (num_crops, 1, height, width); as before
        mask_toSave = mask.float().view(-1, mask.shape[3], mask.shape[4]).unsqueeze(1)
        save_image(
            mask_toSave, img_storage_path + "full_mask.png", nrow=const.NUM_COLUMNS
        )
        if display:
            np_img = plt.imread(img_storage_path + "full_mask.png")
            plt.imshow(np_img)
    # TODO: validate the (un)squeezing is necessary
    crops_masked = crops * mask.unsqueeze(0)
    crops_masked = crops_masked[0, :, :, :, :, :]
    if display:
        display_n_crops(crops_masked, 5)
    if pickle_path != "":
        torch.save(crops_masked, pickle_path)
    return crops_masked


def upscale(img, factor=2, blurred=True):
    """
    Upscale an image by a given factor
    I.e. a 2x2 image will become a 2n x 2n image,
    where n is the factor
    """
    if blurred:
        img_re_dimd = img.unsqueeze(0)
    else:
        img_re_dimd = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
    upscaled = F.interpolate(
        img_re_dimd, scale_factor=factor, mode="bilinear", align_corners=True
    )[0]
    return upscaled


def mean_fluorescences_by_crop(
    const, tif_path, img_storage_path="", display=False, pickle_path=""
):
    crops_masked = clean_mask_n_crop(
        const, tif_path, img_storage_path, display, pickle_path=pickle_path
    )
    mean_fluorescences = torch.mean(crops_masked, dim=(3, 4))
    if pickle_path != "":
        torch.save(mean_fluorescences, pickle_path + "_mean_fluorescences")
    return mean_fluorescences


def mean_fluorescences_by_pixel():
    return


def load_strain_names(
    path_to_plate_layout_excelsheet,
    WTs_savepath="",
    plate_layout_savepath="",
    color_marker_for_WT="FF93C47D",
):
    wb = xl.load_workbook(path_to_plate_layout_excelsheet)
    sheet = wb.active
    strain_name_cells = sheet["B2":"Y15"]
    data = []
    WTs = []
    WT_color = color_marker_for_WT
    for row in strain_name_cells:
        row_values = np.array([cell.value for cell in row])
        row_colors = [cell.fill.fgColor.rgb for cell in row]
        is_WT = [color == WT_color for color in row_colors]
        row_WTs = row_values[is_WT]
        WTs = WTs + row_WTs.tolist()
        data.append(row_values)
    plate_layout_df = pd.DataFrame(data)
    WTs = set(WTs)
    if WTs_savepath != "":
        utils.to_pickle(WTs, WTs_savepath)
    if plate_layout_savepath != "":
        utils.to_pickle(plate_layout_df, plate_layout_savepath)
    return WTs, plate_layout_df


def reassemble_crops(const, crops, num_timesteps):
    """
    Re-assemble the crops into a single image
    """
    img = torch.zeros(
        size=(
            num_timesteps,
            const.NUM_ROWS * const.CELL_WIDTH_X,
            const.NUM_COLUMNS * const.CELL_WIDTH_Y,
        )
    )
    for i in range(const.NUM_ROWS):
        for j in range(const.NUM_COLUMNS):
            img[
                :,
                i * const.CELL_WIDTH_X : (i + 1) * const.CELL_WIDTH_X,
                j * const.CELL_WIDTH_Y : (j + 1) * const.CELL_WIDTH_Y,
            ] = crops[:, i, j]
    return img


def join_strain_IDs_w_param_data(param_data, param_name, strain_IDs, WT_set):
    merged_df = pd.DataFrame(
        columns=["strain", param_name, "WT", "time", "strain_replicate"]
    )
    ts = utils.time_series(param_data)
    n = len(ts)
    strain_replicates = dict()
    for i in range(const.NUM_ROWS):
        for j in range(const.NUM_COLUMNS):
            strain_name = strain_IDs.iloc[i, j]
            if strain_name in strain_replicates:
                strain_replicates[strain_name] += 1
            else:
                strain_replicates[strain_name] = 1
            if strain_name in WT_set:
                WT = True
            else:
                WT = False
            values = param_data[:, i, j].numpy()
            new_subdf = pd.DataFrame(
                {
                    "strain": n * [strain_name],
                    param_name: values,
                    "WT": n * [WT],
                    "time": ts,
                    "strain_replicate": n * [strain_replicates[strain_name]],
                }
            )
            merged_df = pd.concat([merged_df, new_subdf], ignore_index=True)
    return merged_df
