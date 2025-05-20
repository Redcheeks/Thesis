import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Type
from neuron import Neuron, NeuronFactory
from descending_drive import cortical_input
from simulation.models.LIF_model1 import LIF_Model1
from simulation.models.LIF_model2_v3 import LIF_Model2v3
from simulation.models.LIF_model3 import LIF_Model3
from simulation.models.LIF_model3_v2 import LIF_Model3v2

"""
Script for plotting neuron voltage trace, reset voltage (excitation), inhibition traces and Input Current CI
Instructions to use: 
In MAIN - Make sure correct model is uncommented

For details on how plots are made, see the relevant method

"""  ## ------------- Simulation Parameters ------------- ##
T = 1000  # Simulation Time [ms] (Overall time might differ as CI can increase this if needed)
DT = 0.1  # Time step in [ms]
neuron_pool_size = 300  # Total number of Neurons in the pool

## ------ Cortical input Simulation Parameters ------ ##

number_of_clusters = 3  # Number of clusters
max_I = 9  # Max input current (nA)
CCoV = 0  # Cluster-common noise CoV (%)
ICoV = 0  # Independent noise CoV (%)
signal_type = "step-sinusoid"  # Options: "sinusoid.hz" -- "trapezoid" -- "triangular" -- "step-sinusoid" -- "step"
freq = 2  # Frequency for sinusoid


## ------ Neurons to be modelled & plotted. ------ ##
NEURON_INDEXES: List[int] = [10, 20, 30, 45]


def run_model(
    model_class: Type, neurons: List[Neuron], CI: np.ndarray
) -> List[Tuple[Neuron, Tuple[np.ndarray, np.ndarray]]]:
    results = []
    for neuron in neurons:
        Iinj = CI[: int(T / DT), neuron.number]
        output = model_class.simulate_neuron(T, DT, neuron=neuron, Iinj=Iinj)
        # output may have 2 or 3 or 4 elements depending on model
        if model_class == LIF_Model3v2:
            voltage, spikes, inhib, reset = output
            results.append((neuron, (voltage, spikes, inhib, reset)))
        else:
            voltage, spikes, inhib = output
            results.append((neuron, (voltage, spikes, inhib)))
    return results


def plot_voltage_traces(
    simulation_data: List[Tuple[Neuron, Tuple[np.ndarray, np.ndarray]]],
    CI: np.ndarray,
    model_name: str,
) -> None:
    time = np.arange(0, len(CI) * DT, 1)
    import math

    rows = int(np.ceil(np.sqrt(len(simulation_data))))
    cols = int(np.ceil(len(simulation_data) / rows))
    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    fig.suptitle(f"{model_name} - Voltage Traces", fontsize=16, y=0.98)
    axs = axs.flatten()
    for i, (neuron, (v, sp)) in enumerate(simulation_data):
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

        ax.set_ylim([-80, -20])
        ax.axhline(neuron.V_th_mV, color="k", ls="--", label="Threshold")
        ax.set_ylabel("Membrane Potential (mV)")
        ax.set_xlabel("Time (ms)")
        ax.set_title(f"Neuron {neuron.number}")
        ax.legend()
    # Hide unused axes
    for j in range(len(simulation_data), len(axs)):
        fig.delaxes(axs[j])
    # Custom legend for spike types
    from matplotlib.lines import Line2D

    custom_lines = [
        Line2D([0], [0], color="gray", ls="--", alpha=0.4, label="Spike"),
        Line2D([0], [0], color="blue", ls="--", lw=1, alpha=0.7, label="Doublet"),
    ]
    fig.legend(
        handles=custom_lines, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2
    )
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/{model_name}_output.png")


def plot_inhibition_traces(
    simulation_data: List[Tuple[Neuron, Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    CI: np.ndarray,
    model_name: str,
) -> None:
    time = np.arange(0, len(CI) * DT, 1)
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
            ax_v.plot(time, v)
            for spike_time in sp:
                ax_v.axvline(spike_time, color="gray", ls="--", alpha=0.4)
            for j in range(1, len(sp)):
                isi = sp[j] - sp[j - 1]
                if 3 <= isi <= 10:
                    ax_v.axvline(sp[j], color="blue", ls="--", lw=1, alpha=0.7)
            ax_v.axhline(neuron.V_th_mV, color="k", ls="--")
            ax_v.set_ylabel("Membrane Potential (mV)")
            ax_v.set_title(f"Neuron {neuron.number} - Voltage")

            # Inhibition subplot
            ax_i = axs[i][1]
            ax_i.plot(time, inhib, color="orange")
            ax_i.set_ylabel("Inhibition Level")
            ax_i.set_title(f"Neuron {neuron.number} - Inhibition trace")
            for spike_time in sp:
                ax_i.axvline(spike_time, color="gray", ls="--", alpha=0.4)
            for j in range(1, len(sp)):
                isi = sp[j] - sp[j - 1]
                if 3 <= isi <= 10:
                    ax_i.axvline(sp[j], color="blue", ls="--", lw=1, alpha=0.7)
            ax_i.set_xlabel("Time (ms)")

            # Reset voltage subplot
            ax_r = axs[i][2]
            ax_r.plot(time, reset, color="r")
            ax_r.set_ylabel("Reset Voltage (mV)")
            ax_r.set_title(f"Neuron {neuron.number} - Reset Voltage trace")
            for spike_time in sp:
                ax_r.axvline(spike_time, color="gray", ls="--", alpha=0.4)
            for j in range(1, len(sp)):
                isi = sp[j] - sp[j - 1]
                if 3 <= isi <= 10:
                    ax_r.axvline(sp[j], color="blue", ls="--", lw=1, alpha=0.7)
            ax_r.set_xlabel("Time (ms)")
        else:
            v, sp, inhib = values
            # Voltage trace with spikes
            ax_v = axs[i][0]
            ax_v.plot(time, v)
            for spike_time in sp:
                ax_v.axvline(spike_time, color="gray", ls="--", alpha=0.4)
            for j in range(1, len(sp)):
                isi = sp[j] - sp[j - 1]
                if 3 <= isi <= 10:
                    ax_v.axvline(sp[j], color="blue", ls="--", lw=1, alpha=0.7)
            ax_v.axhline(neuron.V_th_mV, color="k", ls="--")
            ax_v.set_ylabel("Membrane Potential (mV)")
            ax_v.set_title(f"Neuron {neuron.number} - Voltage")

            # Inhibition or Boost trace
            ax_i = axs[i][1]
            if model_name == "LIF_Model2v2":
                ax_i.plot(time, inhib, color="r")
                ax_i.set_ylabel("ADP Boost")
                ax_i.set_title(f"Neuron {neuron.number} - Excitability Boost factor")
            elif model_name == "LIF_Model1":
                ax_i.plot(time, inhib, color="orange")
                ax_i.set_ylabel("Inhibition Level")
                ax_i.set_title(f"Neuron {neuron.number} - Inhibition trace")
            else:
                ax_i.plot(time, inhib, color="r")
                ax_i.set_ylabel("Reset Voltage (mV)")
                ax_i.set_title(f"Neuron {neuron.number} - Reset Voltage trace")
            for spike_time in sp:
                ax_i.axvline(spike_time, color="gray", ls="--", alpha=0.4)
            # Add ISI blue lines for doublets
            for j in range(1, len(sp)):
                isi = sp[j] - sp[j - 1]
                if 3 <= isi <= 10:
                    ax_i.axvline(sp[j], color="blue", ls="--", lw=1, alpha=0.7)
            ax_i.set_xlabel("Time (ms)")

    from matplotlib.lines import Line2D

    custom_lines = [
        Line2D([0], [0], color="gray", ls="--", alpha=0.4, label="Spike"),
        Line2D([0], [0], color="blue", ls="--", lw=1, alpha=0.7, label="Doublet"),
    ]
    fig.legend(
        handles=custom_lines, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/{model_name}_inhibition.png")


def plot_cortical_input(CI: np.ndarray):

    plt.figure(1, figsize=(8, 6))
    time = np.arange(0, len(CI) * DT, 1)

    plt.plot(time, CI[:, 50], label=f"Neuron {50}")
    # plt.plot(time, CI[:, 150], label=f"Neuron {150}")
    # plt.plot(time, CI[:, 250], label=f"Neuron {250}")
    plt.axhline(2 / 3 * max_I, color="r", ls="--", alpha=0.4)
    plt.xlabel("Time (ms)")
    plt.ylabel("Current (nA)")
    plt.title(f"Cortical Input for Neuron {50}")

    os.makedirs("figures", exist_ok=True)
    if CCoV > 0 or ICoV > 0:
        plt.savefig(f"figures/CI_{signal_type}_withNoise.png")
    else:
        plt.savefig(f"figures/CI_{signal_type}.png")


if __name__ == "__main__":
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
    # Create neurons with NEURONFACTORY class (# Generate list of new neurons)
    all_neurons = NeuronFactory.create_neuron_pool(
        distribution=True, number_of_neurons=neuron_pool_size
    )
    neurons = [all_neurons[i] for i in NEURON_INDEXES]

    # Run models 1,2 and 3 and plot voltage trace, optional: inhibition, excitation
    for model_class, name in zip(
        [LIF_Model1, LIF_Model2v3, LIF_Model3, LIF_Model3v2],
        ["LIF_Model1", "LIF_Model2v3", "LIF_Model3", "LIF_Model3v2"],
    ):
        results = []
        for neuron in neurons:
            Iinj = CI[: int(T / DT), neuron.number]
            if model_class == LIF_Model3v2 or model_class == LIF_Model3:
                voltage, spikes, inhibition, reset = model_class.simulate_neuron(
                    T, DT, neuron=neuron, Iinj=Iinj
                )
                results.append((neuron, (voltage, spikes, inhibition, reset)))
            else:
                voltage, spikes, inhibition = model_class.simulate_neuron(
                    T, DT, neuron=neuron, Iinj=Iinj
                )
                results.append((neuron, (voltage, spikes, inhibition)))

        # Plot voltage trace as before
        # plot_voltage_traces([(n, (v, s)) for n, (v, s, _) in results], CI, name)

        # Plot inhibition trace
        plot_inhibition_traces(results, CI, name)

    print("Figures generated in the 'figures/' folder.")
    plt.show()
