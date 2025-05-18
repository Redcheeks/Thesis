import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import seaborn as sns

"""
Experimental Data Plotting code
Instructions to use: 
In MAIN - Make sure correct data is imported & uncomment the plots you want created. 

For details on how plots are made, see the relevant method

"""

## ------------- Heatmap Plotting setup ------------- ##
inactive_threshold = 5  # Active
threshold_low = 40  # Orange scaling threshold
threshold_high = 150  # doublet red threshold
force_to_neuron_offset = 0.2  # [seconds] The time delay between the onset of an EMG signal and the measurable force output in muscle contraction, known as the electromechanical delay (EMD), is typically between 30 and 100 milliseconds (ms)


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

    # Plotting the heatmap
    from matplotlib.colors import LinearSegmentedColormap

    # Update colormap for better contrast: pale yellow to warmer orange and red
    cmap = LinearSegmentedColormap.from_list(
        "custom", ["#fffde7", "#ffcc66", "#ff9900", "#cc0000"]
    )

    # Improve color contrast for inactive periods
    cmap.set_bad(color="#d9d9d9")  # light gray for inactivity
    plt.style.use("default")  # light background

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
    ax2.set_ylabel("Force (N)", color="blue")
    ax2.tick_params(axis="y", colors="blue")

    plt.tight_layout()


def force_curve(data, data_name: str):
    # plot experimental force curve

    force = data["force"].flatten()
    time = np.arange(len(force)) / data["fs"].item()

    plt.plot(time, force, color="blue")
    plt.title(f"{data_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")


def _main():

    ## ---------------- Import data ---------------- ##
    trapezoid_sinusoid2hz = scipy.io.loadmat(
        "Experimental Data Analysis/trapezoid10mvc_sinusoid2hz5to15mvc.mat"
    )
    trapezoid = scipy.io.loadmat("Experimental Data Analysis/trapezoid20mvc.mat")

    trapezoid_repetitive = scipy.io.loadmat(
        "Experimental Data Analysis/trapezoid5mvc_repetitive_doublets_SORTED.mat"
    )

    ## ---------------- PLOT FORCE CURVES FOR ALL 3 DATA FILES ---------------- ##
    plt.figure(figsize=(18, 10))
    plt.subplot(3, 1, 1)
    force_curve(trapezoid, "Trapezoid 20% MVC")

    plt.subplot(3, 1, 2)
    force_curve(trapezoid_sinusoid2hz, "Trapezoid 5% MVC & 2hz Sinusoid - 5-15% MVC")

    plt.subplot(3, 1, 3)
    force_curve(trapezoid_repetitive, "Trapezoid 5% MVC")

    plt.subplots_adjust(hspace=0.5, top=0.9)
    plt.suptitle("Experimental Force Curves", fontweight="bold")

    os.makedirs("figures", exist_ok=True)
    plt.savefig(
        f"figures/force_curves.png",
    )

    print("Figure generated in the 'figures/' folder.")

    ## ---------------- PLOT HEATMAP FOR ALL 3 DATA FILES IN SEPERATE PLOTS ---------------- ##
    plt.figure(figsize=(18, 8))
    plt.subplot()
    heatmap(trapezoid)
    plt.subplots_adjust(top=0.83, right=1)
    plt.suptitle("Trapezoid input at 20% MVC", fontweight="bold")
    os.makedirs("figures", exist_ok=True)
    plt.savefig(
        f"figures/experimental_heatmap_trapezoid.png",
    )

    plt.figure(figsize=(18, 8))
    plt.subplot()
    heatmap(trapezoid_repetitive)
    plt.subplots_adjust(top=0.83, right=1)
    plt.suptitle("Trapezoid input at 5% MVC", fontweight="bold")
    os.makedirs("figures", exist_ok=True)
    plt.savefig(
        f"figures/experimental_heatmap_trapezoid_repetitive.png",
    )

    plt.figure(figsize=(18, 8))
    plt.subplot()
    heatmap(trapezoid_sinusoid2hz)
    plt.subplots_adjust(top=0.83, right=1)
    plt.suptitle(
        "Trapezoid input at 10% MVC + 2Hz-Sinus at 5-15%MVC", fontweight="bold"
    )
    os.makedirs("figures", exist_ok=True)
    plt.savefig(
        f"figures/experimental_heatmap_TrapSinus.png",
    )

    ## ----------------  ----------------  ----------------  ---------------- ##
    print("Figures generated in the 'figures/' folder.")
    plt.show()


if __name__ == "__main__":
    _main()
