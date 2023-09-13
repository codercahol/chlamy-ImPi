import pickle
from pathlib import Path
import numpy as np


def to_pickle(obj, path):
    with open(Path(path), "wb") as f:
        pickle.dump(obj, f)


def from_pickle(path):
    with open(Path(path), "rb") as f:
        return pickle.load(f)


def cartesian_to_linear_index(i, j, num_columns):
    """
    Convert from cartesian coordinates to linear index.

    ASSUMES: that counting happens row-wise
        (ie. first row is counted before the second is started)
    Input:
        i: row index
        j: column index
        num_columns: number of columns
    Output:
        linear index
    """
    return i * num_columns + j


def time_series(data):
    """
    Construct a time-series by inference from the shape of the data provided
    Input:
        data: np array of shape (num_timesteps, num_rows, num_columns)
    Output:
        time_series: np array of length (num_timesteps) in units of [hr]'s
    """
    num_timesteps = data.shape[0]
    # ASSUMPTION: 30 minutes per timestep
    ts = np.arange(num_timesteps) * 30 / 60
    return ts


def flatten_format_multiIndex(mi):
    """
    Flatten a pandas multiIndex into a list of strings
    Input:
        mi: pandas multiIndex (list of tuples of strings)
    Output:
        flattened: list of strings
    """
    fmt = lambda x: x[0] if x[1].strip() == "" else "_".join(x).strip()
    return [fmt(col) for col in mi]
