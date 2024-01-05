import datetime
import re

import numpy as np
import pandas as pd


def location_to_index(loc: str) -> tuple[int, int]:
    """Convert a location string, e.g. "A1" or "P12", to a zero-indexed tuple, e.g. (0, 0)"""
    assert 2 <= len(loc) <= 3
    letter = loc[0]
    number = int(loc[1:])

    assert letter in "ABCDEFGHIJKLMNOP"

    i = number - 1
    j = ord(letter) - ord("A")

    return i, j


def index_to_location(i: int, j: int) -> str:
    """Convert a zero-indexed tuple, e.g. (0, 0), to a location string, e.g. "A1" """
    assert 0 <= i <= 16
    assert 0 <= j <= 24

    letter = chr(ord("A") + j)
    number = i + 1

    return f"{letter}{number}"


def index_to_location_rowwise(x):
    """Convert a zero-indexed tuple, e.g. (0, 0), to a location string, e.g. "A1" """

    letter = chr(ord("A") + x.i)
    number = x.j + 1

    return f"{letter}{number}"


def spreadsheet_plate_to_numeric(plate: str) -> int:
    """Convert a plate string, e.g. "Plate 01", to a numeric value, e.g. 1"""
    assert plate.startswith("Plate ")
    return int(plate[6:])


def parse_name(f):
    """Parse the name of a file, e.g. `20200303 7-M4 2h-2h.npy` or `20231119 07-M3 20h ML.npy` or `20231213 9-M5_2h-2h.npy`

    Thank github copilot for this very ugly function. I have added unit tests to make sure it works!
    """
    f = str(f)
    parts = f.split(" ")

    assert len(parts) in {2, 3, 4}, f

    middle = parts[1].split("-")
    plate_num = int(middle[0])

    if len(parts) == 2:
        measurement_num = middle[1].split("_")[0]
    else:
        measurement_num = middle[1]

    if len(parts) == 2:
        time_regime = parts[1].split("_")[1].split(".")[0]
    elif len(parts) == 3:
        assert len(parts[2].split(".")) == 2, f
        time_regime = parts[2].split(".")[0]
    else:
        assert len(parts[3].split(".")) == 2, f
        time_regime = parts[2] + " " + parts[3].split(".")[0]

    assert plate_num in {99, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, f
    assert re.match(r"M[1-6]", measurement_num), f
    assert time_regime in {
        "30s-30s",
        "1min-1min",
        "10min-10min",
        "2h-2h",
        "20h ML",
        "20h HL",
    }, f"{time_regime}, {f}"

    return plate_num, measurement_num, time_regime


def compute_measurement_times(meta_df: pd.DataFrame) -> list[datetime.datetime]:
    """In this function, we compute the time of each y2 or npq measuremnt."""
    meta_df["Datetime"] = meta_df[["Date", "Time"]].apply(
        lambda x: pd.to_datetime(x["Date"]) + pd.to_timedelta(x["Time"]), axis=1
    )

    assert len(meta_df) <= 82
    return meta_df["Datetime"].tolist()
