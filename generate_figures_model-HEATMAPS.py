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
from neuron import NeuronFactory

"""
Model Output Plotting code
Instructions to use: 
In MAIN - Make sure correct model is uncommented

For details on how plots are made, see the relevant method

"""
## ------------- General Simulation Parameters ------------- ##
T = 10000  # Simulation Time [ms]
DT = 0.1  # Time step in [ms]
neuron_pool_size = 300  # Total number of Neurons in the pool
neurons_to_plot = range(10, 40)  # Maximum amount of active neurons to show.
## ------------- Synaptic Input Simulation Parameters ------------- ##

number_of_clusters = 3  # Number of clusters
max_I = 9  # Max input current (nA)
CCoV = 0  # Cluster-common noise CoV (%)
ICoV = 0  # Independent noise CoV (%)
signal_type = "step_sinusoid"  # Options: "sinusoid.hz" -- "trapezoid" -- "triangular" -- "step-sinusoid" -- "step"
freq = 2  # Frequency for sinusoid

## ------------- Plotting setup ------------- ##

inactive_threshold = 5
threshold_high = 100
threshold_low = 30

# Update colormap for better contrast: pale yellow to warmer orange and red
cmap = LinearSegmentedColormap.from_list(
    "custom", ["#fffde7", "#ffcc66", "#ff9900", "#cc0000"]
)
# Improve color contrast for inactive periods
cmap.set_bad(color="#d9d9d9")  # light gray for inactivity
plt.style.use("default")  # light background
## ------------------------------------------------------------ ##


def heatmap(data_to_plot):

    ## ------------------------------------------------------------ ##

    # Extract variables
    discharge_times = data_to_plot  # spike times
    fs = 1 / (DT)  # Sampling frequency in 1/seconds!!

    total_time = int(T * fs)  # Total time

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
            t_end = int(spikes[j + 1]) if j + 1 < len(spikes) else total_time
            if freq < inactive_threshold:
                freq_matrix[i, t_start:t_end] = np.nan

            else:
                freq_matrix[i, t_start:t_end] = freq

        # Mark the period after the last spike as inactive
        freq_matrix[i, int(spikes[-1]) :] = np.nan

    ax = sns.heatmap(
        freq_matrix,
        cmap=cmap,
        vmin=threshold_low,
        vmax=threshold_high,
        cbar=True,
        mask=np.isnan(freq_matrix),
    )

    # Add horizontal lines between neurons
    for y in range(1, freq_matrix.shape[0]):
        ax.axhline(y=y, color="lightgray", linewidth=0.5)

    # Annotate doublets
    for y in range(freq_matrix.shape[0]):
        for x in range(freq_matrix.shape[1]):
            val = freq_matrix[y, x]
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
    ax.set_yticklabels(neurons_to_plot)
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
    ## ---------------- Create Neurons ---------------- ##

    all_neurons = NeuronFactory.create_neuron_pool(
        distribution=True, number_of_neurons=neuron_pool_size
    )
    neurons = [all_neurons[i] for i in neurons_to_plot]

    ## ---------------- PLOT HEATMAP FOR ALL MODELS IN SEPERATE PLOTS ---------------- ##
    # Run for all models
    for model_class, name in zip(
        [LIF_Model1, LIF_Model2v3, LIF_Model3, LIF_Model3v2],
        ["LIF_Model1", "LIF_Model2v3", "LIF_Model3", "LIF_Model3v2"],
    ):
        results = []
        for i, neuron in enumerate(neurons):
            Iinj = CI[: int(T / DT), neuron.number]
            _, spikes, *rest = model_class.simulate_neuron(
                T, DT, neuron=neuron, Iinj=Iinj
            )
            results.append([spikes / DT])  # ms to time step

        plt.figure(figsize=(18, 8))
        plt.subplot()
        heatmap(results)
        plt.subplots_adjust(top=0.83, right=1)
        plt.suptitle(
            f"{name} Output for {max_I} nA (Trapezoid + 2 Hz Sinusoid) input",
            fontweight="bold",
        )
        os.makedirs("figures", exist_ok=True)
        plt.savefig(
            f"figures/heatmap_{name}.png",
        )
        # print(f"{name} done!")
    ## ----------------  ----------------  ----------------  ---------------- ##
    print("Figures generated in the 'figures/' folder.")
    plt.show()


if __name__ == "__main__":
    _main()
