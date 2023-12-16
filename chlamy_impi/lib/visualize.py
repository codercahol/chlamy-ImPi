import matplotlib.pyplot as plt
import numpy as np
from lib import utils
from lib.constants import NUM_ROWS, NUM_COLUMNS


def visualize_strain_param_across_plate(data, name, save_path=""):
    """
    Plot an inferred parameter across the plate.
    Input:
        data: 2D numpy array of shape (num_rows, num_columns)
        name: string name of the parameter
        save_path: string path to save the plot
    Output:
        None (saves plot to save_path)
    """
    p = plt.figure()
    plt.imshow(data, cmap="turbo")
    plt.colorbar(extend="both", shrink=0.6)
    plt.title(f"{name} by Location on 384-plate")
    if save_path == "":
        return p
    else:
        plt.savefig(save_path)


def visualize_strain_param_in_time(
    data,
    name,
    save_path="",
):
    """
    Plot an inferred parameter across time for each strain in the dataset.
    Input:
        data: 3D numpy array of shape (num_timesteps, num_rows, num_columns)
        name: string name of the parameter
        save_path: string path to save the plot
    Output:
        None (saves plot to save_path)
    """
    ts = utils.time_series(data)
    colors = plt.cm.turbo(np.linspace(0, 1, NUM_ROWS))

    p = plt.figure()
    for i in range(NUM_ROWS):
        for j in range(NUM_COLUMNS):
            # randomly color the traces to tell them apart
            plt.plot(ts, data[:, i, j], alpha=0.5, linewidth=0.5, color=colors[i])
    plt.xlabel("Time (hr)")
    plt.ylabel(name)
    if save_path == "":
        return p
    else:
        plt.savefig(save_path)


def plot_strain_param_w_binary_label(data, name, ids, yes_group, save_path=""):
    """
    Plot an inferred parameter across strains with a binary label.
    Input:
        data: 1D numpy array of shape (num_timesteps, num_rows, num_columns)
        name: string name of the parameter
        save_path: string path to save the plot
        ids: plate layout dataframe of the strain ids, shape = (num_rows, num_columns)
        yes_group: string name of the group to be labeled as "in group"
    """
    ts = utils.time_series(data)
    p = plt.figure()
    for i in range(NUM_ROWS):
        for j in range(NUM_COLUMNS):
            strain_name = ids.iloc[i, j]
            if strain_name in yes_group:
                color = "green"
            else:
                color = "gray"
            plt.plot(ts, data[:, i, j], alpha=0.5, linewidth=0.25, color=color)
    plt.xlabel("Time (hr)")
    plt.ylabel(name)
    if save_path == "":
        return p
    else:
        plt.savefig(save_path)
