import numpy as np
import matplotlib.pyplot as plt
import os
import math
from typing import List, Tuple, Type
from neuron import Neuron, NeuronFactory
from descending_drive import cortical_input
from simulation.models.LIF_model1 import LIF_Model1
from simulation.models.LIF_model2_v3 import LIF_Model2v3
from simulation.models.LIF_model3 import LIF_Model3
from simulation.models.LIF_model3_v2 import LIF_Model3v2

"""
Script for plotting:
-- neuron voltage trace
-- reset voltage (excitation)
-- inhibition traces
-- Input Current/Simulated Synaptic Input CI

Instructions to use: 
- Select parameters below
- In MAIN - Make sure correct models are uncommented

For details on how plots are made, see the relevant method

"""
### ------------- What to plot? ------------- ###

# For each model, creates figure with voltage trace, reset voltage and inhibition (if available), saves to figures/..
PLOT_COMPLETE_INHIB = True

# For each model, creates figure with only voltage trace, saves to figures/..
PLOT_VOLTAGE = False

PLOT_CI = False  # Plot Cortical Input for first neuron, for debugging...

# Plots frequency curves for the selected neurons, saves to figures/fre...
PLOT_FREQ = False

### ------------- Simulation PARAMETERS ------------- ###
# Overall time might differ as CI can increase this if needed, see descending_drive.py)
T = 5000  # Simulation Time [ms] (
DT = 0.1  # Time step in [ms]
neuron_pool_size = 300  # Total number of Neurons in the pool

### ------------- Cortical input Simulation PARAMETERS ------------- ###

number_of_clusters = 3  # Number of clusters
max_I = 9  # Max input current (nA)
CCoV = 14  # 14  # Cluster-common noise CoV (%)
ICoV = 5  # 5  # Independent noise CoV (%)
signal_type = "trapezoid"  # Options: "sinusoid.hz" -- "trapezoid" -- "triangular" -- "step-sinusoid" -- "step"
freq = 0.2  # Frequency for sinusoid


### ------------- Neurons & Models to be modelled & plotted. ------------- ###
NEURON_INDEXES: List[int] = [35, 95]
MODELS = [LIF_Model1, LIF_Model2v3, LIF_Model3]
MODEL_NAMES = ["LIF_Model1", "LIF_Model2v3", "LIF_Model3"]


## Colors etc

from matplotlib.lines import Line2D

custom_lines = [
    Line2D([0], [0], color="gray", ls="--", alpha=0.4, label="Spike"),
    Line2D([0], [0], color="blue", ls="--", lw=1, alpha=0.7, label="Doublet"),
]
import matplotlib.pylab as pylab

params = {
    "legend.fontsize": "large",
    "axes.labelsize": "large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "medium",
    "ytick.labelsize": "medium",
}
pylab.rcParams.update(params)


## PARAMETERS FOR FIGURES IN REPORT RESULTS:

# ## ------------- Simulation Parameters ------------- ##
# T = 4000  # Simulation Time [ms] (Overall time might differ as CI can increase this if needed for ramps, see descending_drive.py)
# DT = 0.1  # Time step in [ms]
# neuron_pool_size = 300  # Total number of Neurons in the pool

# ## ------ Cortical input Simulation Parameters ------ ##

# number_of_clusters = 3  # Number of clusters
# max_I = 9  # Max input current (nA)
# CCoV = 14  # 14  # Cluster-common noise CoV (%)
# ICoV = 5  # 5  # Independent noise CoV (%)
# signal_type = "trapezoid"  # Options: "sinusoid.hz" -- "trapezoid" -- "triangular" -- "step-sinusoid" -- "step"
# freq = 0.2  # Frequency for sinusoid


# ## ------ Neurons to be modelled & plotted. ------ ##
# NEURON_INDEXES: List[int] = [35, 80, 95, 150]
# MODELS = zip(
#     [LIF_Model1, LIF_Model2v3, LIF_Model3],
#     ["LIF_Model1", "LIF_Model2v3", "LIF_Model3"],
# )


def run_model(
    model_class: Type, neurons: List[Neuron], CI: np.ndarray
) -> List[Tuple[Neuron, Tuple[np.ndarray, np.ndarray]]]:
    results = []

    for neuron in neurons:
        Iinj = CI[:, neuron.number]
        output = model_class.simulate_neuron(T, DT, neuron=neuron, Iinj=Iinj)
        # output may have 2 or 3 or 4 elements depending on model
        results.append((neuron, (output)))
    return results


def plot_voltage_traces(
    simulation_data: List[Tuple[Neuron, List]],
    CI: np.ndarray,
    model_name: str,
) -> None:
    time = np.linspace(0, len(CI) * DT, len(CI))

    rows = len(simulation_data[0])  # how many neurons to plot for

    fig, axs = plt.subplots(rows, 1, figsize=(6, 4 * rows), squeeze=False)
    fig.suptitle(f"{model_name} - Voltage Traces", fontsize=16, y=0.98)
    axs = axs.flatten()
    for i, (neuron, data) in enumerate(simulation_data):
        v, sp = data[0], data[1]  # We ignore any other returned things
        ax = axs[i]
        ax.plot(time, v, label=f"Neuron {neuron.number}")
        for spike_time in sp:
            ax.axvline(spike_time, color="gray", ls="--", alpha=0.4, label="_nolegend_")
        for j in range(1, len(sp)):
            isi = sp[j] - sp[j - 1]
            if 3 <= isi <= 10:
                ax.axvline(
                    sp[j], color="blue", ls="--", lw=1, alpha=0.7, label="_nolegend_"
                )

        ax.set_ylim([-85, -30])
        ax.axhline(neuron.V_th_mV, color="k", ls="--", label="Threshold")
        ax.set_ylabel("Membrane Potential (mV)")
        ax.set_xlabel("Time (ms)")
        ax.set_title(f"Neuron {neuron.number}")
        ax.legend()

    fig.legend(
        handles=custom_lines, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2
    )
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/output_{signal_type}_{model_name}.png")
    print(f"Voltage figure for {model_name} saved in figures/.. .")


def plot_inhibition_traces(
    simulation_data: List[Tuple[Neuron, List]],
    CI: np.ndarray,
    model_name: str,
) -> None:
    time = np.linspace(0, len(CI) * DT, len(CI))

    # For Model 3, we want 3 columns (Voltage, Inhibition, Reset)
    ncols = 3 if (model_name == "LIF_Model3v2" or model_name == "LIF_Model3") else 2
    figsize = (
        (16, 4 * len(simulation_data)) if ncols == 3 else (12, 4 * len(simulation_data))
    )
    fig, axs = plt.subplots(len(simulation_data), ncols, figsize=figsize)
    fig.suptitle(
        f"{model_name} - Voltage and Inhibition, for Imax = {max_I}, with signal-type = {signal_type}",
        fontsize=16,
        y=0.98,
    )
    if len(simulation_data) == 1:
        axs = [axs]

    for i, data in enumerate(simulation_data):
        neuron = data[0]
        values = data[1]
        if (model_name == "LIF_Model3v2" or model_name == "LIF_Model3") and len(
            values
        ) == 4:
            v, sp, inhib, reset = values
            # Voltage subplot
            ax_v = axs[i][0]
            ax_v.plot(time * 1e-3, v)
            for spike_time in sp:
                ax_v.axvline(spike_time * 1e-3, color="gray", ls="--", alpha=0.4)
            for j in range(1, len(sp)):
                isi = sp[j] - sp[j - 1]
                if 3 <= isi <= 10:
                    ax_v.axvline(sp[j] * 1e-3, color="blue", ls="--", lw=1, alpha=0.7)
            ax_v.axhline(neuron.V_th_mV, color="k", ls="--")
            ax_v.set_ylabel("Membrane Potential (mV)")
            ax_v.set_title(f"Neuron {neuron.number} - Voltage")
            ax_v.set_ylim([-90, -15])
            ax_v.set_xlabel("Time (s)")

            # Inhibition subplot
            ax_i = axs[i][1]
            ax_i.plot(time * 1e-3, inhib, color="orange")
            ax_i.set_ylabel("Inhibition Level")
            ax_i.set_title(f"Neuron {neuron.number} - Inhibition trace")
            for spike_time in sp:
                ax_i.axvline(spike_time * 1e-3, color="gray", ls="--", alpha=0.4)
            for j in range(1, len(sp)):
                isi = sp[j] - sp[j - 1]
                if 3 <= isi <= 10:
                    ax_i.axvline(sp[j] * 1e-3, color="blue", ls="--", lw=1, alpha=0.7)
            ax_i.set_xlabel("Time (s)")

            # Reset voltage subplot
            ax_r = axs[i][2]
            ax_r.plot(time * 1e-3, reset, color="r")
            ax_r.set_ylabel("Reset Voltage (mV)")
            ax_r.set_title(f"Neuron {neuron.number} - Reset Voltage trace")
            for spike_time in sp:
                ax_r.axvline(spike_time * 1e-3, color="gray", ls="--", alpha=0.4)
            for j in range(1, len(sp)):
                isi = sp[j] - sp[j - 1]
                if 3 <= isi <= 10:
                    ax_r.axvline(sp[j] * 1e-3, color="blue", ls="--", lw=1, alpha=0.7)
            ax_r.set_xlabel("Time (s)")
        else:
            v, sp, inhib = values
            # Voltage trace with spikes
            ax_v = axs[i][0]
            ax_v.plot(time * 1e-3, v)
            for spike_time in sp:
                ax_v.axvline(spike_time * 1e-3, color="gray", ls="--", alpha=0.4)
            for j in range(1, len(sp)):
                isi = sp[j] - sp[j - 1]
                if 3 <= isi <= 10:
                    ax_v.axvline(sp[j] * 1e-3, color="blue", ls="--", lw=1, alpha=0.7)
            ax_v.axhline(neuron.V_th_mV, color="k", ls="--")
            ax_v.set_ylabel("Membrane Potential (mV)")
            ax_v.set_title(f"Neuron {neuron.number} - Voltage")
            ax_v.set_ylim([-90, -15])
            ax_v.set_xlabel("Time (s)")

            # Inhibition or Boost trace
            ax_i = axs[i][1]
            if model_name == "LIF_Model2v2":
                ax_i.plot(time * 1e-3, inhib, color="r")
                ax_i.set_ylabel("ADP Boost")
                ax_i.set_title(f"Neuron {neuron.number} - Excitability Boost factor")
            elif model_name == "LIF_Model1":
                ax_i.plot(time * 1e-3, inhib, color="orange")
                ax_i.set_ylabel("Inhibition Level")
                ax_i.set_title(f"Neuron {neuron.number} - Inhibition trace")

            else:
                ax_i.plot(time * 1e-3, inhib, color="r")
                ax_i.set_ylabel("Reset Voltage (mV)")
                ax_i.set_title(f"Neuron {neuron.number} - Reset Voltage trace")
            for spike_time in sp:
                ax_i.axvline(spike_time * 1e-3, color="gray", ls="--", alpha=0.4)
            # Add ISI blue lines for doublets
            for j in range(1, len(sp)):
                isi = sp[j] - sp[j - 1]
                if 3 <= isi <= 10:
                    ax_i.axvline(sp[j] * 1e-3, color="blue", ls="--", lw=1, alpha=0.7)
            ax_i.set_xlabel("Time (s)")

    fig.legend(
        handles=custom_lines, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/inhibition_{signal_type}_{model_name}.png")
    print(f"Voltage + Inhib/Excit. figure for {model_name} saved in figures/.. .")


def plot_cortical_input(CI: np.ndarray):

    plt.figure(1, figsize=(8, 6))
    time = np.linspace(0, len(CI) * DT, len(CI))

    plt.plot(time, CI[:, 50], label=f"Neuron {1}")
    plt.axhline(2 / 3 * max_I, color="r", ls="--", alpha=0.4)
    plt.xlabel("Time (ms)")
    plt.ylabel("Current (nA)")
    plt.title(f"Cortical Input for Neuron {1}")

    os.makedirs("figures", exist_ok=True)
    if CCoV > 0 or ICoV > 0:
        plt.savefig(f"figures/CI_{signal_type}_withNoise.png")
    else:
        plt.savefig(f"figures/CI_{signal_type}.png")

    print("CI / Synaptic Input Figure saved in figures/.. .")


def frequency_plots(
    simulation_data: List[Tuple[Neuron, List]],
    model_name: str,
) -> None:

    for neuron, neuron_data in simulation_data:  # for each neuron
        # plt.close()
        fig, ax = plt.subplots()
        spikes = neuron_data[1]
        fs = 1 / DT

        isi = np.diff(spikes)
        isi_freq = 1 / (isi / fs)
        isi_freq = np.insert(isi_freq, 0, 0)  # first spike frequency is 0

        ax.plot(
            spikes / fs,
            isi_freq,
            "o",
        )
        ax.set_ylabel("Instantaneous Frequency [Hz]")
        ax.set_xlabel("Time [s]")
        ax.set_title(f"Neuron number {neuron.number}")

        os.makedirs("figures/frequency-plots", exist_ok=True)
        plt.savefig(f"figures/frequency-plots/{model_name}_n{neuron.number}.png")

        print(f"{model_name} done")


def _main():
    # Create input
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
    # Create neurons with NEURONFACTORY class (Generates list of new neurons)

    neurons = NeuronFactory.create_neuron_subpool(
        NEURON_INDEXES, distribution=True, number_of_neurons=neuron_pool_size
    )

    #  ------------- Run for all models defined in PARAMETERS ------------- #
    if PLOT_COMPLETE_INHIB or PLOT_VOLTAGE or PLOT_FREQ:

        for model_class, name in zip(MODELS, MODEL_NAMES):

            results = run_model(model_class, neurons, CI)

            # ------------- Plot voltage + inhibition + reset/excitation -------------
            if PLOT_FREQ:
                frequency_plots(results, name)
                plt.close("all")

            if PLOT_COMPLETE_INHIB:
                plot_inhibition_traces(results, CI, name)

            if PLOT_VOLTAGE:
                plot_voltage_traces(results, CI, name)

    if PLOT_CI:
        plot_cortical_input(CI)

    print("All Figures generated in the 'figures/..' folder.")
    plt.show()


if __name__ == "__main__":
    _main()
