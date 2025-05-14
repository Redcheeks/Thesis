from matplotlib.colors import LinearSegmentedColormap
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from simulation.models.LIF_model3 import LIF_Model3
from descending_drive import cortical_input
from neuron import NeuronFactory

""" 

Script to produce Heatmap-plots for Experimental data and Model 3 

4 subplots
top: Heatmaps to show active periods and doublets for experimental (left) and model3 (right) 
bottom: Force curve for experimental (left) and synaptic input current/cortical input for model3 (right) 

"""

## ------------- Simulation Parameters ------------- ##
T = 35000  # Simulation Time [ms]
DT = 0.1  # Time step in [ms]
neuron_pool_size = 30  # Total number of Neurons in the pool

## ------ Cortical input Simulation Parameters ------ ##

number_of_clusters = 3  # Number of clusters
max_I = 9  # Max input current (nA)
CCoV = 5  # Cluster-common noise CoV (%)
ICoV = 5  # Independent noise CoV (%)
signal_type = "step-sinusoid"  # Options: "sinusoid.hz" -- "trapezoid" -- "triangular" -- "step-sinusoid" -- "step"
freq = 2  # Frequency for sinusoid

## ----- Plotting setup ------ ##
# Load experimental data
trapezoid_sinusoid2hz = scipy.io.loadmat(
    "Experimental Data Analysis/trapezoid10mvc_sinusoid2hz5to15mvc.mat"
)
trapezoid = scipy.io.loadmat("Experimental Data Analysis/trapezoid20mvc.mat")
## -- CHOOSE WHICH DATASET TO PLOT: --
data = trapezoid_sinusoid2hz

threshold_low = 40  # yellow threshold
threshold_high = 150  # doublet red threshold

# Update colormap for better contrast: pale yellow to warmer orange and red
cmap = LinearSegmentedColormap.from_list(
    "custom", ["#fffde7", "#ffcc66", "#ff9900", "#cc0000"]
)
# Improve color contrast for inactive periods
cmap.set_bad(color="#d9d9d9")  # light gray for inactivity
plt.style.use("default")  # light background


def plot_experimental_heatmap():
    # create heatmap from experimental data
    discharge_times = data["discharge_times"]
    fs = data["fs"].item()  # Sampling frequency

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
    with np.errstate(all="ignore"):
        downsampled_matrix = np.nanmax(
            freq_matrix[:, : num_windows * window_size].reshape(
                len(discharge_times[0]), num_windows, window_size
            ),
            axis=2,
        )

    padded_freq_matrix = downsampled_matrix

    ax = sns.heatmap(
        padded_freq_matrix,
        cmap=cmap,
        vmin=threshold_low,
        vmax=threshold_high,
        # xticklabels=1000,
        # yticklabels=[f" Neuron {i}" for i in range(len(padded_freq_matrix))],
        cbar=True,
        mask=np.isnan(padded_freq_matrix),
    )

    # Add horizontal lines between neurons
    for y in range(1, padded_freq_matrix.shape[0]):
        ax.axhline(y=y, color="lightgray", linewidth=0.5)

    # Add colorbar label
    cbar = ax.collections[0].colorbar
    cbar.set_label("Instantaneous Frequency (Hz)", color="black")

    # Annotate doublets
    for y in range(downsampled_matrix.shape[0]):
        for x in range(downsampled_matrix.shape[1]):
            val = downsampled_matrix[y, x]
            if not np.isnan(val) and val > 150:
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
    xtick_locs = np.arange(0, padded_freq_matrix.shape[1], fs // 5)
    xtick_labels = [f"{int((x * window_size) / fs)}" for x in xtick_locs]
    ax.set_xticks(xtick_locs)
    ax.set_xticklabels(xtick_labels)
    plt.xlabel("Time (s)")
    plt.title("Experimental - Active Periods and Doublets")
    plt.ylabel("Neuron Index")


def plot_model3_heatmap(CI):
    # Generate Model 3 heatmap

    # Create neurons and run Model 3

    all_neurons = NeuronFactory.create_neuron_pool(
        distribution=True, number_of_neurons=neuron_pool_size
    )
    neurons = [
        all_neurons[i] for i in range(len(all_neurons))
    ]  # Using all 30 neurons for heatmap. For larger group dont want all of them.

    model = LIF_Model3()

    # Create a time-aligned frequency matrix
    freq_matrix = np.full((len(neurons), int(T / DT)), np.nan)
    fs = data["fs"].item()  # Use same sampling frequency..

    for i, neuron in enumerate(neurons):
        _, spikes, _, _ = model.simulate_neuron(T, DT, neuron, CI[:, i])
        if len(spikes) < 2:
            continue  # Not enough spikes to calculate frequency
        else:
            # Compute instantaneous frequency using interspike intervals
            isi = np.diff(spikes * 1e-3)  # Convert to seconds
            freq = np.concatenate(([0], 1 / isi))
            for j in range(len(isi)):
                t_start = int(spikes[j])
                t_end = int(spikes[j + 1]) if j + 1 < len(spikes) else int(T / DT)
                if t_end > t_start:
                    freq_matrix[i, t_start:t_end] = freq[j]

    window_size = int(fs * 0.05)  # 50 ms window
    num_windows = int(
        (T / 1000) * fs / window_size
    )  # Ensure it matches the experimental scale

    # Aggregate using max pooling (highlight doublets)
    with np.errstate(all="ignore"):
        downsampled_matrix = np.nanmax(
            freq_matrix[:, : num_windows * window_size].reshape(
                len(neurons), num_windows, window_size
            ),
            axis=2,
        )

    ax = sns.heatmap(
        downsampled_matrix,
        cmap=cmap,
        vmin=threshold_low,
        vmax=threshold_high,
        # xticklabels=1000,
        # linewidths=0.3,
        # linecolor="white",
        cbar=True,
        mask=np.isnan(downsampled_matrix),
    )

    # plt.imshow(spike_matrix, aspect="auto", cmap="hot", interpolation="none")

    # Add horizontal lines between neurons
    for y in range(1, downsampled_matrix.shape[0]):
        ax.axhline(y=y, color="lightgray", linewidth=0.5)

    # Annotate doublets
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
    xtick_locs = np.arange(0, downsampled_matrix.shape[1], int(fs / 10))
    xtick_labels = [f"{int((x * window_size) / fs)}" for x in xtick_locs]
    ax.set_xticks(xtick_locs)
    ax.set_xticklabels(xtick_labels)
    plt.title("Model 3 - Active Periods and Doublets")
    plt.xlabel("Time (s)")
    plt.ylabel("Neuron Index")


def plot_experimental_force():
    # plot experimental force curve

    force = data["force"].flatten()
    time = np.arange(len(force)) / data["fs"].item()

    plt.plot(time, force, color="blue")
    plt.title("Experimental Force Curve")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")


def plot_model3_synaptic_input(CI):
    # Generate synaptic input (cortical input) for Model 3
    time = np.linspace(0, T / 1000, int(T / DT))  # time in seconds

    plt.plot(time, CI[:, 0], color="orange")
    plt.title("Model 3 - Synaptic Input")
    plt.xlabel("Time (s)")
    plt.ylabel("Synaptic Input (nA)")


def generate_report_figure():

    CI = cortical_input(
        neuron_pool_size,
        number_of_clusters,
        max_I,
        T,
        DT,
        CCoV,
        ICoV,
        signal_type,
        freq,
    )

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plot_experimental_heatmap()

    plt.subplot(2, 2, 2)
    plot_model3_heatmap(CI)

    plt.subplot(2, 2, 3)
    plot_experimental_force()

    plt.subplot(2, 2, 4)
    plot_model3_synaptic_input(CI)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/experimental_vs_model_heatmaps.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    generate_report_figure()
