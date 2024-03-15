import logging

import numpy as np

logger = logging.getLogger(__name__)


def fix_erroneous_time_points(meta_df, img_array, names):
    """In this function, we handle issues with the fluorescence images. Occasionally there is an error with the data
    (possibly caused by a battery failure for the saturating pulse), which is recorded in the .csv meta data files.
    We need to locate and remove these single frames, and then re-align the time points to the frames by removing the
    appropriate frame from the image stack.
    """
    # One possible column layout in the csv files
    if set(meta_df.columns) == {
        "Date",
        "Time",
        "No.",
        "PAR",
        "F1",
        "F2",
        "F3",
        "Fm'1",
        "Fm'2",
        "Fm'3",
        "Y(II)1",
        "Y(II)2",
        "Y(II)3",
    }:
        # Look for cases where Fm'1 == Fm'2 == Fm'3 == 0, as these signify failed measurements
        meta_df["failed_measurement"] = (
            (meta_df["Fm'1"] == 0.0)
            & (meta_df["Fm'2"] == 0.0)
            & (meta_df["Fm'3"] == 0.0)
        )
    elif set(meta_df.columns) == {
        "Date",
        "Time",
        "No.",
        "PAR",
        "Y(II)1",
        "Y(II)2",
        "Y(II)3",
        "NPQ1",
        "NPQ2",
        "NPQ3",
    }:
        # Look for cases where Y(II) == 0 and NPQ == 1, as these signify failed measurements
        meta_df["failed_measurement"] = (
            (meta_df["Y(II)1"] == 0.0)
            & (meta_df["Y(II)2"] == 0.0)
            & (meta_df["Y(II)3"] == 0.0)
            & (meta_df["NPQ1"] == 1.0)
            & (meta_df["NPQ2"] == 1.0)
            & (meta_df["NPQ3"] == 1.0)
        )
    elif set(meta_df.columns) == {"Date", "Time", "No.", "PAR", "F1", "Fm'1", "Y(II)1"}:
        # Look for cases where Fm'1 == 0 and Y(II)1 == 0, as these signify failed measurements
        meta_df["failed_measurement"] = (meta_df["Fm'1"] == 0.0) & (
            meta_df["Y(II)1"] == 0.0
        )
    elif set(meta_df.columns) == {
        "Date",
        "Time",
        "No.",
        "PAR",
        "Y(II)1",
        "Y(II)2",
        "Y(II)3",
    }:
        meta_df["failed_measurement"] = (
            (meta_df["Y(II)1"] == 0.0)
            & (meta_df["Y(II)2"] == 0.0)
            & (meta_df["Y(II)3"] == 0.0)
        )
    # TODO - validate this
    elif set(meta_df.columns) == {
        "Date",
        "Time",
        "No.",
        "PAR",
        "F1",
        "F2",
        "F3",
        "Fm'1",
        "Fm'2",
        "Fm'3",
        "Fo'1",
        "Fo'2",
        "Fo'3",
        "Y(II)1",
        "Y(II)2",
        "Y(II)3",
        "Y(NPQ)1",
        "Y(NPQ)2",
        "Y(NPQ)3",
        "Y(NO)1",
        "Y(NO)2",
        "Y(NO)3",
        "NPQ1",
        "NPQ2",
        "NPQ3",
        "qN1",
        "qN2",
        "qN3",
        "qP1",
        "qP2",
        "qP3",
        "qL1",
        "qL2",
        "qL3",
        "ETR1",
        "ETR2",
        "ETR3",
        "Abs.1",
        "Abs.2",
        "Abs.3",
        "Inh.1",
        "Inh.2",
        "Inh.3",
    }:
        meta_df["failed_measurement"] = (
            (meta_df["Y(II)1"] == 0.0)
            & (meta_df["Y(II)2"] == 0.0)
            & (meta_df["Y(II)3"] == 0.0)
        )
    else:
        print(meta_df.columns)
        raise NotImplementedError

    failed_rows = meta_df[meta_df["failed_measurement"]]
    assert len(failed_rows) <= 3

    for i, (ind, row) in enumerate(failed_rows.iterrows()):
        logger.warning(f"Found failed measurement at index {ind}. Attempting to fix!")
        logger.warning(f"Bad row: {row}")

        meta_df = meta_df.drop(ind)
        bad_frame_index = (ind - i) * 2
        img_array = np.delete(img_array, bad_frame_index, axis=2)

    assert (
        img_array.shape[2] % 2 == 0
    ), f"Odd number of time steps ({img_array.shape[2]}) in {names}"
    assert (
        img_array.shape[2] == len(meta_df) * 2
    ), f"Number of frames ({img_array.shape[2]}) does not match number of rows in meta_df ({len(meta_df)}) in {names}"

    return meta_df, img_array


def remove_repeated_initial_frame(img_array) -> np.array:
    """Check for a repetition of the first pair of frames"""
    if np.all(img_array[:, :, 0, ...] == img_array[:, :, 2, ...]) and np.all(
        img_array[:, :, 3, ...] == img_array[:, :, 1, ...]
    ):
        logger.warning("Found repeated initial frame pair. Removing!")
        img_array = img_array[:, :, 2:, ...]

    return img_array


def remove_repeated_initial_frame_tif(tif: np.array) -> np.array:
    """As above, but for raw tif files which have time step in the first dimension"""
    if np.all(tif[0, ...] == tif[2, ...]) and np.all(tif[3, ...] == tif[1, ...]):
        logger.warning("Found repeated initial frame pair. Removing!")
        tif = tif[2:, ...]

    return tif


def remove_failed_photos(tif):
    """
    Remove photos that are all black

    Input:
        tif: numpy array (num_images, height, width)
    Output:
        tif: numpy array of shape (num_images, height, width)
    """
    max_per_timestep = tif.max(1).max(1)
    keep_image = max_per_timestep > 0

    logger.warning(
        f"Discarding {sum(~keep_image)} images (indices {list(np.argwhere(~keep_image))}) which are all black"
    )

    tif = tif[keep_image]
    return tif
