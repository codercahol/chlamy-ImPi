# %% load packages

%load_ext autoreload
%autoreload 2

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from lib import utils
from lib.constants import NUM_ROWS, NUM_COLUMNS
import lib.visualize as viz
import lib.inference as inf
import lib.data_processing as dap
import lib.data_juggling as dj
import torch
import pandas as pd

# %% Infer photosynthetic parameters
# mean_fluor = utils.from_pickle("../output/intensities_3h.pkl")
# or
mean_fluor = torch.load("../output/2023-03-23_3hr_mean_fluor.pkl")
(QEY, PY, NPQ, Y_NPQ) = inf.compute_photosynthetic_params(mean_fluor)

# %% Visualize the results

viz.visualize_strain_param_across_plate(
    QEY, "Quantum Electron Yield", "../output/QEY.png"
)
viz.visualize_strain_param_in_time(PY, "Photosynthetic Yield", "../output/PY.png")
viz.visualize_strain_param_in_time(NPQ, "NPQ", "../output/NPQ.png")
viz.visualize_strain_param_in_time(Y_NPQ, "Y(NPQ)", "../output/Y_NPQ.png")

# %% load in strain names and WT's

xl_path = "../data/plate_strain_IDs.xlsx"
WTs_savepath = "../output/WT_set.pkl"
names_savepath = "../output/strain_names.pkl"
WTs, plate_layout_df = dap.load_strain_names(xl_path, WTs_savepath, names_savepath)

# %% plot photosynthetic yield with WT's labelled
viz.plot_strain_param_w_binary_label(
    PY,
    "Photosynthetic Yield",
    plate_layout_df,
    WTs,
    save_path="../output/PY_WTs_labelled.png",
)

# %% Merge strain names and WT's with photosynthetic yield data

PY_df = dap.join_strain_IDs_w_param_data(PY, "PY", plate_layout_df, WTs)
# average across strain replicates
PY_df = dj.average_across_strain_replicates(PY_df, "PY")
PY_df

# %%
# graph with WT's labelled
# tool tip with different mutants
# number of failed cell tiles -> go/no-go if the colony picking machine didn't work
