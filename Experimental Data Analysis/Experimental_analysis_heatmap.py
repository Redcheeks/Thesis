import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import seaborn as sns

## Heatmap Plotting code written with help of ChatGPT


def _main():
    # Import data
    trapezoid_sinusoid2hz = scipy.io.loadmat(
        "Experimental Data Analysis/trapezoid10mvc_sinusoid2hz5to15mvc.mat"
    )
    trapezoid = scipy.io.loadmat("Experimental Data Analysis/trapezoid20mvc.mat")

    ## ---------------- CHOOSE WHICH DATA TO PLOT! ---------------- ##

    data_to_plot = trapezoid
    # Plotting parameters
    threshold_low = 40  # yellow threshold
    threshold_high = 150  # doublet red threshold

    ## ------------------------------------------------------------ ##

    # Extract variables
    discharge_times = data_to_plot["discharge_times"]
    fs = data_to_plot["fs"].item()  # Sampling frequency

    # Determine the total time of the experiment in samples
    all_spike_times = np.concatenate(
        [neuron.flatten() for neuron in discharge_times[0]]
    )
    total_time = int(np.ceil(np.max(all_spike_times))) + 1

    # Create a time-aligned frequency matrix
    freq_matrix = np.full((len(discharge_times[0]), total_time), np.nan)

    for i, neuron_data in enumerate(discharge_times[0]):
        spikes = np.sort(neuron_data.flatten())
        isi = np.diff(spikes)
        isi_freq = 1 / (isi / fs)

        for j in range(len(isi)):
            t_start = int(spikes[j])
            t_end = int(spikes[j + 1]) if j + 1 < len(spikes) else total_time
            if t_end > t_start:
                freq_matrix[i, t_start:t_end] = isi_freq[j]

    window_size = int(fs * 0.05)  # 50 ms window

    num_windows = freq_matrix.shape[1] // window_size

    # Aggregate using max pooling (highlight doublets)
    downsampled_matrix = np.nanmax(
        freq_matrix[:, : num_windows * window_size].reshape(
            len(discharge_times[0]), num_windows, window_size
        ),
        axis=2,
    )

    padded_freq_matrix = downsampled_matrix

    from matplotlib.colors import LinearSegmentedColormap

    # Update colormap for better contrast: pale yellow to warmer orange and red
    cmap = LinearSegmentedColormap.from_list(
        "custom", ["#fffde7", "#ffcc66", "#ff9900", "#cc0000"]
    )

    # Improve color contrast for inactive periods
    cmap.set_bad(color="#d9d9d9")  # light gray for inactivity
    plt.style.use("default")  # light background

    plt.figure(figsize=(18, 6))
    ax = sns.heatmap(
        padded_freq_matrix,
        cmap=cmap,
        vmin=threshold_low,
        vmax=threshold_high,
        xticklabels=1000,
        yticklabels=[f"Neuron {i}" for i in range(len(padded_freq_matrix))],
        # linewidths=0.3,
        # linecolor="white",
        cbar=True,
        mask=np.isnan(padded_freq_matrix),
    )

    # Add horizontal lines between neurons
    for y in range(1, padded_freq_matrix.shape[0]):
        ax.axhline(y=y, color="lightgray", linewidth=0.5)

    # Add colorbar label
    cbar = ax.collections[0].colorbar
    cbar.set_label("Instantaneous Frequency (Hz)", color="black")

    # # Optional: Add vertical reference lines for event markers
    # event_times_sec = [10, 20, 30]  # seconds
    # for t_sec in event_times_sec:
    #     bin_index = int((t_sec * fs) / window_size)
    #     ax.axvline(x=bin_index, color="cyan", linestyle="--", linewidth=0.8)

    # Update axis text colors for light background
    ax.tick_params(colors="black")
    ax.yaxis.label.set_color("black")
    ax.xaxis.label.set_color("black")

    # Annotate doublets using downsampled matrix
    for y in range(downsampled_matrix.shape[0]):
        for x in range(downsampled_matrix.shape[1]):
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

    # Adjust x-axis tick labels to use integer seconds
    xtick_locs = np.arange(0, padded_freq_matrix.shape[1], 20)
    xtick_labels = [f"{int((x * window_size) / fs)}" for x in xtick_locs]
    ax.set_xticks(xtick_locs)
    ax.set_xticklabels(xtick_labels)
    plt.xlabel("Time (s)")

    plt.title(
        f"Neurons with Instantaneous Frequency\nYellow > {threshold_low} Hz, Red > {threshold_high} Hz (Doublets)"
    )
    plt.ylabel("Neuron Index")
    # plt.subplots_adjust(top=1.0, bottom=0.1)  # Adjusted spacing between title and plot

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#d9d9d9", edgecolor="k", label="Inactive"),
        Patch(facecolor="#fffde7", edgecolor="k", label="Normal spiking"),
        Patch(facecolor="#cc0000", edgecolor="k", label="Doublet (D)"),
    ]

    # Move legend closer to the title
    plt.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),  # carefully tuned just below title
        ncol=3,
        frameon=False,
    )

    # Save the figure
    import os

    data_filename = (
        "trapezoid10mvc_sinusoid2hz5to15mvc"
        if data_to_plot is trapezoid_sinusoid2hz
        else "trapezoid20mvc"
    )
    os.makedirs("figures", exist_ok=True)
    plt.savefig(
        f"figures/experimental_ifreq_heatmap_{data_filename}.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.show()


if __name__ == "__main__":
    _main()
