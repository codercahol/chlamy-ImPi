import logging

import numpy as np

logger = logging.getLogger(__name__)


def fix_erroneous_time_points(meta_df, img_array):
    """In this function, we handle issues with the fluorescence images. Occasionally there is an error with the data
    (possibly caused by a battery failure for the saturating pulse), which is recorded in the .csv meta data files.
    We need to locate and remove these single frames, and then re-align the time points to the frames by removing the
    appropriate frame from the image stack.
    """
    # One possible column layout in the csv files
    if np.all(meta_df.columns == ["Date", "Time", "No.", "PAR", "F1", "F2", "F3", "Fm'1", "Fm'2", "Fm'3", "Y(II)1", "Y(II)2", "Y(II)3"]):

        # Look for cases where Fm'1 == Fm'2 == Fm'3 == 0, as these signify failed measurements
        meta_df["failed_measurement"] = (meta_df["Fm'1"] == 0.) & (meta_df["Fm'2"] == 0.) & (meta_df["Fm'3"] == 0.)
        failed_rows = meta_df[meta_df["failed_measurement"]]

        assert len(failed_rows) <= 3

        for i, (ind, row) in enumerate(failed_rows.iterrows()):
            logger.warning(f"Found failed measurement at index {ind}. Attempting to fix!")

            meta_df = meta_df.drop(ind)
            bad_frame_index = (ind - i) * 2
            img_array = np.delete(img_array, bad_frame_index, axis=2)

    else:
        raise NotImplementedError

    assert img_array.shape[2] % 2 == 0, f"Odd number of time steps ({img_array.shape[2]})"
    assert img_array.shape[2] == len(meta_df) * 2, f"Number of frames ({img_array.shape[2]}) does not match number of rows in meta_df ({len(meta_df)})"

    return meta_df, img_array


def remove_repeated_initial_frame(img_array) -> np.array:
    """Check for a repetition of the first pair of frames
    """
    if (np.all(img_array[:, :, 0, ...] == img_array[:, :, 2, ...]) and
        np.all(img_array[:, :, 3, ...] == img_array[:, :, 1, ...])):
            logger.warning("Found repeated initial frame pair. Removing!")
            img_array = img_array[:, :, 2:, ...]

    return img_array