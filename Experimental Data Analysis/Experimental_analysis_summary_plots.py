import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import seaborn as sns
import matplotlib.pylab as pylab
from matplotlib.colors import LinearSegmentedColormap


"""
Experimental Data Plotting Script
Instructions to use: 
- Choose parameters below, make sure correct data is imported
 
For details on how plots are made, see the relevant method

"""

### ------------- Plotting PARAMETERS ------------- ###

PLOT_FORCE_CURVE = True  # Plot force curves, saves figure
FORCE_OVERLAY = True  # Force curve overlay on heatmap, (Figure name reflects this)
PLOT_HEATMAPS = True  # Plot heatmaps in individual plots, saves figures

# For heatmap:
inactive_threshold = 5  # Active
threshold_low = 40  # Orange scaling threshold
threshold_high = 100  # doublet red threshold
# [seconds] The time delay between the onset of an EMG signal and the measurable force output in muscle contraction, known as the electromechanical delay (EMD), is typically between 30 and 100 milliseconds (ms)
force_to_neuron_offset = -0.2

### ---------------- IMPORT DATA ---------------- ###

trap_sinus = scipy.io.loadmat(
    "Experimental Data Analysis/trapezoid10mvc_sinusoid2hz5to15mvc.mat"
)
trapezoid = scipy.io.loadmat("Experimental Data Analysis/trapezoid20mvc_fixForce.mat")

trap_repet = scipy.io.loadmat(
    "Experimental Data Analysis/trapezoid5mvc_repetitive_doublets_sorted.mat"
)
# Choose which data to plot and its title!##
data_set = [trap_sinus, trapezoid, trap_repet]
data_names = [
    "Trapezoid 10% MVC + 2 Hz Sinusoid 5-15% MVC",
    "Trapezoid 20% MVC",
    "Trapezoid 5% MVC",
]

## ------------- Plotting colors ------------- ##

# Update colormap for better contrast: pale yellow to warmer orange and red
cmap = LinearSegmentedColormap.from_list(
    "custom", ["#fffde7", "#ffcc66", "#ff9900", "#cc0000"]
)

# Improve color contrast for inactive periods
cmap.set_bad(color="#d9d9d9")  # light gray for inactivity
plt.style.use("default")  # light background

## ------------- Plotting text-sizes ------------- ##
params = {
    "legend.fontsize": "medium",
    "axes.labelsize": "large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
    "figure.titlesize": "xx-large",
}
pylab.rcParams.update(params)


def heatmap(data_to_plot):

    ## ------------------------------------------------------------ ##

    # Extract variables
    discharge_times = data_to_plot["discharge_times"]
    fs = data_to_plot["fs"].item()  # Sampling frequency

    # Determine the total time of the experiment using the force curve
    force_curve = data_to_plot["force"].flatten()
    total_time = len(force_curve)  # Total number of samples in the force curve

    # Create a time-aligned frequency matrix
    freq_matrix = np.full((len(discharge_times[0]), total_time), np.nan)

    # Calculate ISI frequencies and fill the matrix
    for i, neuron_data in enumerate(discharge_times[0]):
        spikes = np.sort(neuron_data.flatten())
        if len(spikes) < 2:
            continue  # Not enough spikes to calculate frequency

        isi = np.diff(spikes)
        isi_freq = 1 / (isi / fs)

        # Fill the frequency matrix for active periods, and breaks
        for j, freq in enumerate(isi_freq):
            t_start = int(spikes[j])
            t_end = int(spikes[j + 1]) if j + 1 < len(spikes) else total_time
            if freq < inactive_threshold:
                freq_matrix[i, t_start:t_end] = np.nan

            else:
                freq_matrix[i, t_start:t_end] = freq

        # Mark the period after the last spike as inactive (this shouldnt be necessary...)
        freq_matrix[i, int(spikes[-1]) :] = np.nan

    # Downsample the frequency matrix (50 ms windows)
    window_size = int(fs * 0.05)

    num_windows = total_time // window_size
    downsampled_matrix = np.nanmax(
        freq_matrix[:, : num_windows * window_size].reshape(
            len(discharge_times[0]), num_windows, window_size
        ),
        axis=2,
    )

    ax = sns.heatmap(
        downsampled_matrix,
        cmap=cmap,
        vmin=threshold_low,
        vmax=threshold_high,
        yticklabels=[
            f"Neuron {num}" if i % 2 != 0 else ""
            for i, num in enumerate(range(1, len(downsampled_matrix) + 1))
        ],
        cbar=True,
        mask=np.isnan(downsampled_matrix),
    )

    # Add horizontal lines between neurons
    for y in range(1, downsampled_matrix.shape[0]):
        ax.axhline(y=y, color="lightgray", linewidth=0.5)

    # Add colorbar label
    cbar = ax.collections[0].colorbar
    cbar.set_label("Instantaneous Frequency (Hz)", color="black")

    # Annotate doublets
    for y, row in enumerate(downsampled_matrix):
        for x, val in enumerate(row):
            val = downsampled_matrix[y, x]
            if not np.isnan(val) and val > threshold_high:
                ax.text(
                    x + 0.5,
                    y + 0.5,
                    "D",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=6,
                )

    # X-axis tick labels (adjusted to time in seconds)
    xtick_locs = np.arange(0, downsampled_matrix.shape[1], fs // 20)
    xtick_labels = [f"{int((x * window_size) / fs)}" for x in xtick_locs]

    ax.set_xticks(xtick_locs)
    ax.set_xticklabels(xtick_labels)
    plt.xlabel("Time (s)")

    plt.title(
        f"Neurons with Instantaneous Frequency\n Pale Yellow < {threshold_low} Hz, Red > {threshold_high} Hz (Doublets)",
        pad=30,
    )
    plt.ylabel("Neuron Index")

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor="#d9d9d9",
            edgecolor="k",
            label=f"Inactive (<{inactive_threshold}Hz)",
        ),
        Patch(facecolor="#fffde7", edgecolor="k", label="Normal spiking"),
        Patch(facecolor="#cc0000", edgecolor="k", label="Doublet (D)"),
    ]

    # Move legend closer to the title
    plt.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),  # move legend above plot, below title
        ncol=3,
        frameon=False,
    )
    ax.invert_yaxis()

    ## OPTIONAL: ------------ Overlay force curve on secondary y-axis
    if FORCE_OVERLAY:
        ax2 = ax.twinx()
        time = np.linspace(
            force_to_neuron_offset * 0.01 * fs,
            downsampled_matrix.shape[1] + force_to_neuron_offset * 0.01 * fs,
            total_time,
        )
        ax2.plot(
            time,
            force_curve[:total_time],
            color="blue",
            linewidth=0.5,
            alpha=0.7,
        )
        ax2.set_ylabel("Force (% MVC)", color="blue")
        ax2.tick_params(axis="y", colors="blue")

    plt.tight_layout()


def force_curve(data, data_name: str):
    # plot experimental force curve

    force = data["force"].flatten()
    time = np.arange(len(force)) / data["fs"].item()

    plt.plot(time, force, "k")
    plt.title(f"{data_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (% MVC)")


def _main():

    ## ---------------- PLOT FORCE CURVES FOR ALL DATA FILES ---------------- ##
    if PLOT_FORCE_CURVE:
        plt.figure(figsize=(10, 8))
        plt.suptitle("Experimental Force Curves", fontweight="bold")

        for index, [data, name] in enumerate(zip(data_set, data_names)):
            plt.subplot(len(data_set), 1, index + 1)
            force_curve(data, name)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5, top=0.9)

        os.makedirs("figures", exist_ok=True)
        plt.savefig(
            f"figures/force_curves.png",
        )

        print("Force-Curve figure generated in the 'figures/' folder.")

    ## ---------------- PLOT HEATMAP FOR ALL 3 DATA FILES IN SEPERATE PLOTS ---------------- ##
    for data, name in zip(data_set, data_names):
        plt.figure(figsize=(15, 7))
        plt.subplot()
        heatmap(data)
        plt.subplots_adjust(top=0.83, right=1.07)
        plt.suptitle(f"Neuron firing for Experiment - {name}", fontweight="bold")

        os.makedirs("figures", exist_ok=True)
        if FORCE_OVERLAY:
            plt.savefig(
                f"figures/experimental_heatmap_{name}_withForce.png",
            )
        else:
            plt.savefig(
                f"figures/experimental_heatmap_{name}.png",
            )
    ## ----------------  ----------------  ----------------  ---------------- ##

    print("Figures generated in the 'figures/' folder.")
    plt.show()


if __name__ == "__main__":
    _main()
