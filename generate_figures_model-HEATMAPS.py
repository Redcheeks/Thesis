from matplotlib.colors import LinearSegmentedColormap
import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import seaborn as sns
from simulation.models.LIF_model1 import LIF_Model1
from simulation.models.LIF_model2_v3 import LIF_Model2v3
from simulation.models.LIF_model3 import LIF_Model3
from simulation.models.LIF_model3_v2 import LIF_Model3v2
from descending_drive import cortical_input
from neuron import Neuron, NeuronFactory

"""
Model Output Plotting code
Instructions to use: 
In MAIN - Make sure correct model is uncommented

For details on how plots are made, see the relevant method

"""
## ------------- General Simulation Parameters ------------- ##
T = 10000  # Simulation Time [ms] (Overall time might differ as CI can increase this if needed)
DT = 0.1  # Time step in [ms]
neuron_pool_size = 300  # Total number of Neurons in the pool
neurons_to_plot = np.linspace(
    1, 110, 30, dtype=int
)  # range(50, 100)  # Neurons to show. Selection of neurons in a range
MODELS = zip(
    [LIF_Model2v3, LIF_Model3],
    ["LIF_Model2v3", "LIF_Model3"],
)
## ------------- Synaptic Input Simulation Parameters ------------- ##

number_of_clusters = 3  # Number of clusters
max_I = 7  # Max input current (nA)
CCoV = 14  # 14  # Cluster-common noise CoV (%)
ICoV = 5  # 5  # Independent noise CoV (%)
signal_type = "step-sinusoid"  # Options: "sinusoid.hz" -- "trapezoid" -- "triangular" -- "step-sinusoid" -- "step"
freq = 0.2  # Frequency for sinusoid

## ------------- Plotting setup ------------- ##
SHOW_CI = False
SHOW_VOLTAGE = False
inactive_threshold = 4
threshold_high = 100
threshold_low = 5

# Update colormap for better contrast: pale yellow to warmer orange and red
cmap = LinearSegmentedColormap.from_list(
    "custom", ["#fffde7", "#ffcc66", "#ff9900", "#cc0000"]
)
# Improve color contrast for inactive periods
cmap.set_bad(color="#d9d9d9")  # light gray for inactivity
plt.style.use("default")  # light background
## ------------------------------------------------------------ ##


def heatmap(data_to_plot, time):

    ## ------------------------------------------------------------ ##

    # Extract variables
    discharge_times = data_to_plot  # spike times
    fs = 1 / (DT)  # Sampling frequency in 1/seconds!!

    total_time = int(time / fs)  # Total time

    # Create a time-aligned frequency matrix
    freq_matrix = np.full((len(discharge_times), total_time), np.nan)

    # Calculate ISI frequencies and fill the matrix
    for i, neuron_data in enumerate(discharge_times):
        spikes = neuron_data[0]
        if len(spikes) < 2:
            continue  # Not enough spikes to calculate frequency
        isi = np.diff(spikes * DT)  # [ms]
        isi_freq = 1 / (isi) * 1e3

        # Fill the frequency matrix for active periods, and breaks
        for j, freq in enumerate(isi_freq):
            t_start = int(spikes[j])
            t_end = int(spikes[j + 1]) if j + 1 < len(spikes) else int(spikes[j])
            if freq < inactive_threshold:
                freq_matrix[i, t_start:t_end] = np.nan

            else:
                freq_matrix[i, t_start:t_end] = freq

        # Mark the period after the last spike as inactive
        freq_matrix[i, int(spikes[-1]) :] = np.nan

    # # Downsample the frequency matrix (200 ms windows)
    # window_size = int(200)

    # num_windows = total_time // window_size
    # downsampled_matrix = np.nanmax(
    #     freq_matrix[:, : num_windows * window_size].reshape(
    #         len(discharge_times[0]), num_windows, window_size
    #     ),
    #     axis=2,
    # )
    downsampled_matrix = freq_matrix

    ax = sns.heatmap(
        downsampled_matrix,
        cmap=cmap,
        vmin=threshold_low,
        vmax=threshold_high,
        yticklabels=neurons_to_plot,
        cbar=True,
        mask=np.isnan(downsampled_matrix),
    )

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

    # X-axis tick labels (adjusted to time in seconds)
    xtick_locs = np.arange(0, total_time, total_time / 20)
    xtick_labels = [f"{int((x) / fs)*1e-3}" for x in xtick_locs]

    ax.set_xticks(xtick_locs)
    ax.set_xticklabels(xtick_labels)
    plt.xlabel("Time (s)")

    plt.title(
        f"Neurons with Instantaneous Frequency\n Pale Yellow < {threshold_low} Hz, Red > {threshold_high} Hz (Doublets)",
        pad=30,
    )
    plt.ylabel("Neuron Index")
    # ax.set_yticklabels(neurons_to_plot)
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
    ax.invert_yaxis()

    # Move legend closer to the title
    plt.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),  # move legend above plot, below title
        ncol=3,
        frameon=False,
    )


def plot_CI(CI):
    plt.figure(1, figsize=(8, 6))
    time = np.linspace(0, len(CI) * DT, len(CI))

    plt.plot(time, CI[:, 50], label=f"Neuron {50}")
    plt.axhline(2 / 3 * max_I, color="r", ls="--", alpha=0.4)
    plt.xlabel("Time (ms)")
    plt.ylabel("Current (nA)")
    plt.title(f"Cortical Input for Neuron {50}")


def plot_voltage(CI, neuron: Neuron):
    plt.figure(2, figsize=(8, 6))

    v, sp, _ = LIF_Model2v3.simulate_neuron(
        T, DT, neuron=neuron, Iinj=CI[:, neuron.number]
    )
    time = np.linspace(0, len(CI) * DT, len(CI))
    plt.plot(time, v, label=f"Neuron {neuron.number}")

    for spike_time in sp:
        plt.axvline(spike_time, color="gray", ls="--", alpha=0.4, label="_nolegend_")
    for j in range(1, len(sp)):
        isi = sp[j] - sp[j - 1]
        if 3 <= isi <= 10:
            plt.axvline(
                sp[j], color="blue", ls="--", lw=1, alpha=0.7, label="_nolegend_"
            )

    plt.ylim([-80, -20])
    plt.axhline(neuron.V_th_mV, color="k", ls="--", label="Threshold")
    plt.ylabel("Membrane Potential (mV)")
    plt.xlabel("Time (ms)")
    plt.title(f"Voltage trace for model2v3 for Neuron {neuron.number}")
    plt.legend()


def _main():

    ## ---------------- Create Synaptic Drive ---------------- ##
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
    if SHOW_CI:
        plot_CI(CI)

    ## ---------------- Create Neurons ---------------- ##

    all_neurons = NeuronFactory.create_neuron_pool(
        distribution=True, number_of_neurons=neuron_pool_size
    )
    neurons = [all_neurons[i] for i in neurons_to_plot]

    if SHOW_VOLTAGE:
        plot_voltage(CI, neurons[20])

    ## ---------------- PLOT HEATMAP FOR ALL MODELS IN SEPERATE PLOTS ---------------- ##
    # Run for all models
    for model_class, name in MODELS:
        results = []
        for i, neuron in enumerate(neurons):
            Iinj = CI[:, neuron.number]
            _, spikes, *rest = model_class.simulate_neuron(
                T, DT, neuron=neuron, Iinj=Iinj
            )
            results.append([spikes / DT])  # ms to time step

        plt.figure(figsize=(18, 8))
        plt.subplot()
        heatmap(results, time=len(Iinj) / DT)
        plt.subplots_adjust(top=0.83, right=1)
        plt.suptitle(
            f"{name} Output for {max_I} nA {signal_type} input",
            fontweight="bold",
        )
        os.makedirs("figures", exist_ok=True)
        plt.savefig(
            f"figures/heatmap_{name}_{signal_type}.png",
        )
        # print(f"{name} done!")
    ## ----------------  ----------------  ----------------  ---------------- ##
    print("Figures generated in the 'figures/' folder.")
    plt.show()


if __name__ == "__main__":
    _main()
