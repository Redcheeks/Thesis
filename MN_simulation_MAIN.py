from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from neuron import NeuronFactory, Neuron  # dataclass used for neuron parameters.
from simulation.models.LIF_model1 import LIF_Model1  # class handling LIF models.
from simulation.models.LIF_model_SIMPLE import LIF_SIMPLE  # class handling LIF models.
from descending_drive import cortical_input  # script for creating current input.
import seaborn as sns


T = 1000  # Simulation Time [ms] (Overall time might differ as CI can increase this if needed)
DT = 0.1  # Time step in [ms]


def Output_plots(CI: np.array, simulation_data: List[Tuple[Neuron]]) -> None:
    """Produces a membrane potential - Time plot for the given simulation results."""

    time = np.linspace(0, np.shape(CI)[0] * DT, np.shape(CI)[0])
    import math

    rows = int(np.ceil(np.sqrt(len(simulation_data))))
    cols = int(np.ceil(len(simulation_data) / rows))
    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    fig.suptitle("Voltage Traces - {}".format(LIF_Model1.__name__), fontsize=16)
    axs = axs.flatten()
    for i, neuron_data_pair in enumerate(simulation_data):
        neuron, (v, sp, trace) = neuron_data_pair
        ax = axs[i]
        ax.plot(time, v, label=f"Neuron {neuron.number}")
        for spike_time in sp:
            spike_index = int(spike_time / DT)
            if 0 <= spike_index < len(v):
                pass  # v[spike_index] += 20  # show a voltage spike
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
    fig.legend(handles=custom_lines, loc="upper center", ncol=2)


def Freq_inst_plot(CI: np.array, simulation_data: List[Tuple[Neuron]]) -> None:

    freq = {}  # record freq over time for each neuron
    fig, ax = plt.subplots()
    for neuron_data_pair in simulation_data:
        v, sp, trace = neuron_data_pair[1]

        if len(sp) < 2:
            freq = 0  # Not enough spikes to compute frequency
        else:
            # Compute interspike intervals, time from ms to seconds
            isi = np.diff(sp * 1e-3)

            # First spike has no frequency - set to 0, this also makes array same length as sp.
            # Assign frequency values to corresponding spike times (excluding the first spike)
            freq[neuron_data_pair[0].number] = np.concatenate(
                ([0], 1 / isi)
            )  # First spike has no frequency

            ax.plot(sp, freq[neuron_data_pair[0].number], ".")

    ax.set_ylabel("Frequency (HZ)")
    ax.set_xlabel("Time (ms)")
    ax.set_title("Frequency-time Plot")
    ax.legend([neuron_data_pair[0].number for neuron_data_pair in simulation_data])


def output_heatmat(CI: np.array, simulation_data: List[Tuple[Neuron]]) -> None:

    time = np.linspace(0, np.shape(CI)[0] * DT, np.shape(CI)[0])
    outputs = []
    for neuron_data_pair in simulation_data:
        v, sp = neuron_data_pair[1]
        # if sp.size:
        #     sp_num = (sp / DT).astype(int) - 1
        #     v[sp_num] += 20  # draw nicer spikes
        outputs.append(v)

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        outputs,
        cmap="coolwarm",
        xticklabels=1000,
        yticklabels=[
            neuron_data_pair[0].number for neuron_data_pair in simulation_data
        ],
    )

    plt.xlabel("Time (steps)")
    plt.ylabel("Neurons")
    plt.title("Neuron Output Heatmap")


def _main():

    ## ------------- Simulation Parameters ------------- ##

    # T = 1000  # Simulation Time [ms]
    # DT = 0.1  # Time step in [ms]
    neuron_pool_size = 300  # Total number of Neurons in the pool

    ## SELECT THE MODEL TO RUN

    model_choice = LIF_SIMPLE  # Options: LIF_SIMPLE, LIF_Model1, LIF_Model2, LIF_Model3

    ## -- Cortical input - simulation parameters -- ##

    number_of_clusters = 1  # Number of clusters
    max_I = 15  # Max input current (nA)
    CCoV = 20  # Cluster-common noise CoV (%)
    ICoV = 5  # Independent noise CoV (%)
    signal_type = "trapezoid"  # Options:  "sinusoid.hz" -- "trapezoid" -- "triangular" -- "step-sinusoid" -- "step"
    freq = 2  # Frequency for sinusoid

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
    neuron_indexes = [50, 100, 250]  # Neurons to be modelled & plotted.
    neuron_indexes = [
        50,
    ]

    neurons_to_simulate = [all_neurons[i] for i in neuron_indexes]
    # neurons_to_simulate = all_neurons # OPTION: Use this to model all 300 neurons!

    ##-------------------------------------------------- ##

    # TODO Run simulation for selected neurons with iunput CI (possible make a new method in SIMULATION class to run many simulations?)

    # Create a list of (neuron-object , (v, sp) ) data-pairs
    simulation_results = [
        (
            b,
            model_choice.simulate_neuron(
                T, DT, neuron=b, Iinj=CI[: int(T / DT), b.number]
            ),
        )
        for b in neurons_to_simulate
    ]

    # TODO Some Plotting choices

    # Output_plots(CI, simulation_results)
    # Freq_inst_plot(CI, simulation_results)
    # output_heatmat(CI, simulation_results)

    ISI = np.diff(simulation_results[0][1][1]) * 1e3  # to seconds

    mean_ISI = ISI.mean(axis=0)
    std_ISI = ISI.std(axis=0)
    CoV_ISI = std_ISI / mean_ISI
    print(f"mean ISI = {mean_ISI}")
    print(f"std ISI = {std_ISI}")
    print(f"mean CoV of ISI % = {np.mean(CoV_ISI)*100}")

    plt.show()


if __name__ == "__main__":
    _main()
