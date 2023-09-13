import streamlit as st
from lib.utils import from_pickle
from lib.constants import NUM_ROWS, NUM_COLUMNS
import lib.visualize as viz
import lib.inference as inf
import lib.data_processing as dap
import lib.data_juggling as dj
import pandas as pd
import torch

import altair as alt


mean_fluor = torch.load("output/2023-03-23_3hr_mean_fluor.pkl")
(QEY, PY, NPQ, Y_NPQ) = inf.compute_photosynthetic_params(mean_fluor)

xl_path = "data/plate_strain_IDs.xlsx"
WTs_savepath = "output/WT_set.pkl"
names_savepath = "output/strain_names.pkl"
WTs, plate_layout_df = dap.load_strain_names(xl_path, WTs_savepath, names_savepath)


PY_df = pd.DataFrame(columns=["strain", "intensities", "WT"])

"""
# Photosynthetic Yield
Labelled by WT
"""


PY_df = dap.join_strain_IDs_w_param_data(PY, "PY", plate_layout_df, WTs)
PY_df2 = dj.average_across_strain_replicates(PY_df, "PY")
# there are 8 NaN strains with blank cells
# 3 are unknown, 4 are DW15.C1, 1 is ca-1
# there are also 5 unfilled wells that were later labelled
# and there is the one blank well in the top left corner
# there should be multiple blank wells throughout the plate
# for more accurate/robust inference


# TODO - change to checkboxes? for multiple selections?
# or use dash instead of streamlit + altair

# add CI's
# anticipation stuff / reading for bkgnd

# TODO - estimate base-error rate & sanity check Fv/Fm
strains = PY_df["strain"].unique().tolist()
strain_dropdown = alt.binding_select(
    options=strains + [None], name="Strain: ", labels=strains + ["All"]
)
strain_select = alt.selection_point(
    fields=["strain"], empty="all", bind=strain_dropdown
)
color = alt.condition(
    strain_select,
    alt.Color("WT:N").scale(range=["#e41a1c", "#49ecf6"]),
    alt.value("lightgray"),
)
opac = alt.condition(strain_select, alt.value(1.0), alt.value(0.3))

base = (
    alt.Chart(PY_df)
    .mark_line()
    .encode(
        x="time",
        y="mean(PY)",
        color=color,
        size=alt.condition(strain_select, alt.value(2), alt.value(0.3)),
        detail="strain",
        tooltip=["strain", "mean(PY)"],
        opacity=opac,
    )
    .interactive()
    .properties(width=800, height=600)
)
filter_strains = base.add_params(strain_select)
error_color = alt.value("#b300b3")
error_opac = alt.condition(strain_select, alt.value(0.4), alt.value(0.0))
band = (
    alt.Chart(PY_df)
    .mark_errorband(extent="ci")
    .encode(
        x="time",
        y=alt.Y("mean(PY)"),
        color=error_color,
        detail="strain",
        opacity=error_opac,
    )
)
band + filter_strains

# deprecated
strains2 = PY_df2["strain"].unique().tolist()
strain_dropdown2 = alt.binding_select(
    options=strains2 + [None], name="Strain: ", labels=strains2 + ["All"]
)
strain_select2 = alt.selection_point(
    fields=["strain"], empty="all", bind=strain_dropdown2
)
color2 = alt.condition(
    strain_select2,
    alt.Color("WT:N").scale(range=["#e41a1c", "#49ecf6"]),
    alt.value("lightgray"),
)
opac2 = alt.condition(strain_select2, alt.value(1.0), alt.value(0.3))

base2 = (
    alt.Chart(PY_df2)
    .mark_line()
    .encode(
        x="time",
        y="PY_mean",
        color=color2,
        size=alt.condition(strain_select2, alt.value(2), alt.value(0.3)),
        detail="strain",
        tooltip=["strain", "PY_mean"],
        opacity=opac2,
    )
    .interactive()
    .properties(width=800, height=600)
)
filter_strains2 = base2.add_params(strain_select2)
# the original way I did it
# filter_strains2
