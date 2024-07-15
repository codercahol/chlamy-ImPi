import logging

import numpy as np
import pandas as pd

from chlamy_impi.database_creation.constants import get_possible_frame_numbers
from chlamy_impi.error_correction.spurious_frame_detection import detect_spurious_frames

logger = logging.getLogger(__name__)


def fix_spurious_frames(meta_df, img_array, base_filename: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attempt to automatically fix spurious frames in the image array and metadata DataFrame.

    Note: at present, this function often fails and will raise an AssertionError
    """

    bad_df_inds, bad_frame_inds = detect_spurious_frames(meta_df, img_array, base_filename)
    bad_inds = list(zip(bad_df_inds, bad_frame_inds))

    assert len(bad_inds) < 4, f"Found too many erroneous frames ({len(bad_inds)}) for {base_filename}"

    for meta_df_index, frame_num in bad_inds:
        logger.warning(f'Fixing erroneous frame {frame_num} for {base_filename}')

        meta_df = meta_df.drop(meta_df.index[meta_df_index])

        img_array = np.delete(img_array, frame_num, axis=2)

    assert img_array.shape[2] in get_possible_frame_numbers(), f"Unexpected number of frames ({img_array.shape[2]}) for {base_filename}"
    assert img_array.shape[2] == len(meta_df) * 2, f"Number of frames ({img_array.shape[2]}) does not match number of rows in meta_df ({len(meta_df)})"

    return meta_df, img_array
