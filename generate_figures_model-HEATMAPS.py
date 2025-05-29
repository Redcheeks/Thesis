from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pylab as pylab
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


"""
Model Output Plotting code
Instructions to use: 
- Choose parameters below
- In _main() - Make sure correct model is uncommented 

For details on how plots are made, see the relevant method

"""

### ------------- What to Plot ------------- ###

SHOW_HEATMAPS = True  # Plot heatmap for each model, Plots saved to figures/..
SHOW_CI = False  # Plot figure with the synaptic input (Plot not saved)
SHOW_VOLTAGE = False  # Plot figure with voltage curve (Plot not saved)
SHOW_NEURON_NUMBER = 20  # Neuron to plot CI & voltage curve for

DOUBLE_HEATMAP = False  # 2 Heatmaps for 2 different CI shapes in one plot. first shape picked here, 2nd is step-sinusoid (can be changed in main)
### ------------- General Simulation PARAMETERS ------------- ###

T = 25000  # Simulation Time [ms] (Overall time might differ as CI can increase this if needed)
DT = 0.1  # Time step in [ms]
neuron_pool_size = 300  # Total number of Neurons in the pool

# NEURONS to show in heatmap. Selection of neurons in a maximum range between 1 and neuron_pool_size
NEURON_INDEXES = np.linspace(1, 90, 30, dtype=int)

# MODELS to plot heatmaps for
MODELS = [LIF_Model1, LIF_Model2v3, LIF_Model3]
MODEL_NAMES = ["LIF_Model1", "LIF_Model2v3", "LIF_Model3"]

### ------------- Synaptic Input Simulation PARAMETERS ------------- ###

number_of_clusters = 3  # Number of clusters
max_I = 7  # Max input current (nA)
CCoV = 14  # 14  # Cluster-common noise CoV (%)
ICoV = 5  # 5  # Independent noise CoV (%)
signal_type = "step-sinusoid"  # Options: "sinusoid.hz" -- "trapezoid" -- "triangular" -- "step-sinusoid" -- "step"
freq = 0.2  # Frequency for sinusoid

### ------------- Heatmap PARAMETERS ------------- ###

inactive_threshold = 4  # 4Hz is the sweetspot to not loose activity in sinus vally's
threshold_high = 100
threshold_low = inactive_threshold


## ------------- Plotting colors and text-sizes ------------- ##
# Update colormap for better contrast: pale yellow to warmer orange and red
cmap = LinearSegmentedColormap.from_list(
    "custom", ["#fffde7", "#ffcc66", "#ff9900", "#cc0000"]
)
# Improve color contrast for inactive periods
cmap.set_bad(color="#d9d9d9")  # light gray for inactivity
plt.style.use("default")  # light background

params = {
    "legend.fontsize": "medium",
    "axes.labelsize": "large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "medium",
    "ytick.labelsize": "medium",
    "figure.titlesize": "x-large",
}
pylab.rcParams.update(params)
## ------------------------------------------------------------ ##


def heatmap(data_to_plot, time, cbar: bool):

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
        yticklabels=NEURON_INDEXES,
        cbar=cbar,
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
    xtick_labels = [f"{int((x) / fs*1e-3)}" for x in xtick_locs]

    ax.set_xticks(xtick_locs)
    ax.set_xticklabels(xtick_labels)
    plt.xlabel("Time (s)")

    plt.ylabel("Neuron Index")
    # ax.set_yticklabels(NEURON_INDEXES)

    ax.invert_yaxis()


def heatmap_legends():
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor="#d9d9d9",
            edgecolor="k",
            label=f"Inactive (<{inactive_threshold}Hz)",
        ),
        Patch(
            facecolor="#fffde7",
            edgecolor="k",
            label=f"Normal spiking (>{inactive_threshold}Hz",
        ),
        Patch(
            facecolor="#cc0000", edgecolor="k", label=f"Doublet (D)(>{threshold_high}Hz"
        ),
    ]
    # Move legend closer to the title
    plt.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),  # move legend above plot, below title
        ncol=3,
        frameon=False,
    )


def plot_CI(CI):

    time = np.linspace(0, len(CI) * DT, len(CI))

    plt.plot(time, CI[:, SHOW_NEURON_NUMBER], label=f"Neuron {1}")
    if signal_type == "step-sinusoid":
        plt.axhline(2 / 3 * max_I, color="r", ls="--", alpha=0.4)
    else:
        plt.axhline(max_I, color="r", ls="--", alpha=0.4)
    plt.xlabel("Time (ms)")
    plt.ylabel("Current (nA)")
    plt.title(f"Cortical Input for Neuron {1}")


def plot_voltage(CI, neuron: Neuron):
    plt.figure(2, figsize=(10, 8))
    time = np.linspace(0, len(CI) * DT, len(CI))
    ## ---- Plot input ---- ##
    plt.subplot(len(MODELS) + 1, 1, 1)
    plot_CI(CI)
    ## ---- Plot voltage trace for each model ---- ##
    for index, (model_class, name) in enumerate(zip(MODELS, MODEL_NAMES)):
        plt.subplot(len(MODELS) + 1, 1, index + 2)

        results = model_class.simulate_neuron(
            T, DT, neuron=neuron, Iinj=CI[:, neuron.number]
        )
        v, sp = results[0], results[1]

        plt.plot(time, v, label=f"Neuron {neuron.number}")

        for spike_time in sp:
            plt.axvline(
                spike_time, color="gray", ls="--", alpha=0.4, label="_nolegend_"
            )
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
        plt.title(f"Voltage trace for Model {name}, Neuron {neuron.number}")
        plt.tight_layout()
    # plt.legend()


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

    CI_sinus = cortical_input(
        neuron_pool_size,
        number_of_clusters,
        max_I,
        T,
        DT,
        CCoV,
        ICoV,
        "step-sinusoid",
        freq,
    )
    ## ---------------- Plot Synaptic Input ---------------- ##
    if SHOW_CI:
        plt.figure(1, figsize=(8, 6))
        plot_CI(CI)

    ## ---------------- Create Neurons ---------------- ##

    # all_neurons = NeuronFactory.create_neuron_pool(
    #     distribution=True, number_of_neurons=neuron_pool_size
    # )
    # neurons = [all_neurons[i] for i in NEURON_INDEXES]

    neurons = NeuronFactory.create_neuron_subpool(
        NEURON_INDEXES, distribution=True, number_of_neurons=neuron_pool_size
    )

    ## ---------------- Plot voltages for 1 chosen neuron, for all neurons ---------------- ##

    if SHOW_VOLTAGE:
        plot_voltage(CI, neurons[SHOW_NEURON_NUMBER])

    ## ---------------- PLOT HEATMAP FOR ALL MODELS IN SEPERATE PLOTS ---------------- ##
    # Run for all models
    if SHOW_HEATMAPS and DOUBLE_HEATMAP:
        for model_class, name in zip(MODELS, MODEL_NAMES):
            results = []
            for i, neuron in enumerate(neurons):
                Iinj = CI[:, neuron.number]
                _, spikes, *rest = model_class.simulate_neuron(
                    T, DT, neuron=neuron, Iinj=Iinj
                )
                results.append([spikes / DT])  # ms to time step
            results_2 = []
            for i, neuron in enumerate(neurons):
                Iinj = CI_sinus[:, neuron.number]
                _, spikes, *rest = model_class.simulate_neuron(
                    T, DT, neuron=neuron, Iinj=Iinj
                )
                results_2.append([spikes / DT])  # ms to time step

            fig, axs = plt.subplots(2, 2, figsize=(14, 7))
            plt.suptitle(
                f"{name} firing behavior for I = {max_I} nA, shape = {signal_type}, CCoV = {CCoV}%, ICCoV ={ICoV}% ",
                fontweight="bold",
            )
            plt.subplots_adjust(top=0.83)

            plt.subplot(2, 1, 1)
            heatmap(results, time=len(Iinj) / DT, cbar=False)
            heatmap_legends()
            # plt.title(
            #     f"Neurons with Instantaneous Frequency\n Pale Yellow < {threshold_low} Hz, Red > {threshold_high} Hz (Doublets)",
            #     pad=30,
            # )

            plt.subplot(2, 1, 2)
            heatmap(results_2, time=len(Iinj) / DT, cbar=False)

            cmappable = ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
            fig.colorbar(cmappable, ax=axs[:, :], location="right")

            os.makedirs("figures", exist_ok=True)
            plt.savefig(
                f"figures/heatmap_{name}_double-signal.png",
            )
            print(f"{name} done!")
        ## ----------------  ----------------  ----------------  ---------------- ##
        print("Figures generated in the 'figures/' folder.")

    elif SHOW_HEATMAPS:
        for model_class, name in zip(MODELS, MODEL_NAMES):
            results = []
            for i, neuron in enumerate(neurons):
                Iinj = CI[:, neuron.number]
                _, spikes, *rest = model_class.simulate_neuron(
                    T, DT, neuron=neuron, Iinj=Iinj
                )
                results.append([spikes / DT])  # ms to time step

            plt.figure(figsize=(14, 7))
            plt.subplot()
            heatmap(results, time=len(Iinj) / DT, cbar=True)
            heatmap_legends()
            plt.subplots_adjust(top=0.83, right=1.07)
            plt.suptitle(
                f"{name} firing behavior for I = {max_I} nA, shape = {signal_type}, CCoV = {CCoV}%, ICCoV ={ICoV}% ",
                fontweight="bold",
            )
            plt.title(
                f"Neurons with Instantaneous Frequency",
                pad=30,
            )
            if CCoV or ICoV > 0:
                os.makedirs("figures", exist_ok=True)
                plt.savefig(
                    f"figures/heatmap_{name}_{signal_type}.png",
                )
            else:
                os.makedirs("figures", exist_ok=True)
                plt.savefig(
                    f"figures/heatmap_{name}_{signal_type}_NoNoise.png",
                )
            # print(f"{name} done!")
        ## ----------------  ----------------  ----------------  ---------------- ##
        print("Figures generated in the 'figures/' folder.")

    plt.show()


if __name__ == "__main__":
    _main()
