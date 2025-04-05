from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from neuron import NeuronFactory, Neuron  # dataclass used for neuron parameters.
from simulation.models.LIF_model1 import LIF_Model1  # class handling LIF models.
from simulation.models.LIF_model2 import LIF_Model2  # class handling LIF models.
from simulation.models.LIF_model_SIMPLE import LIF_SIMPLE  # class handling LIF models.
from descending_drive import cortical_input  # script for creating current input.
import seaborn as sns


T = 1000  # Simulation Time [ms]
DT = 0.1  # Time step in [ms]


def Output_plots(CI: np.array, simulation_data: List[Tuple[Neuron,]]):
    """Produces a membrane potential - Time plot for the given simulation results."""

    time = np.linspace(0, np.shape(CI)[0] * DT, np.shape(CI)[0])

    fig, ax = plt.subplots()
    for neuron_data_pair in simulation_data:
        v, sp = neuron_data_pair[1]
        if sp.size:
            sp_num = (sp / DT).astype(int) - 1
            v[sp_num] += 20  # draw nicer spikes

        ax.plot(time, v, "-")

    ax.set_ylim([-80, -20])
    ax.axhline(
        simulation_data[0][0].V_th_mV, color="k", ls="--"
    )  # gets the first neurons threshold level
    ax.set_ylabel("Membrane Potential (mV)")
    ax.set_xlabel("Time (ms)")
    ax.set_title("Neuron Output")
    ax.legend([neuron_data_pair[0].number for neuron_data_pair in simulation_data])


def Freq_inst_plot(CI: np.array, simulation_data: List[Tuple[Neuron,]]):

    freq = {}  # record freq over time for each neuron
    fig, ax = plt.subplots()
    for neuron_data_pair in simulation_data:
        v, sp = neuron_data_pair[1]

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


def output_heatmat(CI: np.array, simulation_data: List[Tuple[Neuron,]]):

    time = np.linspace(0, np.shape(CI)[0] * DT, np.shape(CI)[0])
    outputs = []
    for neuron_data_pair in simulation_data:
        v, sp = neuron_data_pair[1]
        if sp.size:
            sp_num = (sp / DT).astype(int) - 1
            v[sp_num] += 20  # draw nicer spikes
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

    model_choice = LIF_Model2  # Options: LIF_SIMPLE, LIF_Model1, LIF_Model2

    ## -- Cortical input - simulation parameters -- ##

    number_of_clusters = 1  # Number of clusters
    max_I = 7  # Max input current (nA)
    CCoV = 0  # Cluster-common noise CoV (%)
    ICoV = 0  # Independent noise CoV (%)
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
        5,
        50,
        100,
        120,
        150,
        200,
        299,
    ]  # Neurons to be modelled & plotted.

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

    Output_plots(CI, simulation_results)
    Freq_inst_plot(CI, simulation_results)
    # output_heatmat(CI, simulation_results)

    plt.show()


if __name__ == "__main__":
    _main()
