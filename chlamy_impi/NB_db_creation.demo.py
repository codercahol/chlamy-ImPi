# %%
%load_ext autoreload
%autoreload 2

from itertools import product
from pathlib import Path
import logging
import sqlite3

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from chlamy_impi.database_creation.utils import (
    location_to_index,
    parse_name,
    spreadsheet_plate_to_numeric,
)
from chlamy_impi.lib.fv_fm_functions import compute_all_fv_fm_averaged
from chlamy_impi.lib.mask_functions import compute_threshold_mask
from chlamy_impi.lib.npq_functions import compute_all_npq_averaged
from chlamy_impi.lib.y2_functions import compute_all_y2_averaged

logger = logging.getLogger(__name__)

DEV_MODE = False
INPUT_DIR = Path("../output/image_processing/v6/img_array")
IDENTITY_SPREADSHEET_PATH = Path(
    "../data/Identity plates in Burlacot Lab 20231221 simplified.xlsx - large-lib_rearray2.txt.csv"
)
OUTPUT_DIR = Path("./../output/database_creation/v1")
