from typing import List
import numpy as np
import matplotlib.pyplot as plt
from neuron import NeuronFactory, Neuron  # dataclass used for neuron parameters.
from simulation.models.LIF_model1 import LIF_Model1  # class handling LIF models.
from descending_drive import cortical_input  # script for creating current input.


T = 1000  # Simulation Time [ms]
DT = 0.1  # Time step in [ms]


def Output_plots(CI: np.array, simulation_data: List):
    """Produces a membrane potential - Time plot for the given simulation results."""

    time = np.linspace(0, np.shape(CI)[0], np.shape(CI)[0])

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
    ax.legend()


def _main():

    ## ------------- Simulation Parameters ------------- ##

    T = 1000  # Simulation Time [ms]
    DT = 0.1  # Time step in [ms]
    neuron_pool_size = 300  # Total number of Neurons in the pool

    neuron_indexes = [5, 50, 150, 200, 275]  # Neurons to be modelled & plotted.

    ## SELECT THE MODEL TO RUN

    model_choice = LIF_Model1  # Options: LIF_Model1

    ## -- Cortical input - simulation parameters -- ##

    number_of_clusters = 5  # Number of clusters
    max_I = 40  # Max input current (nA)
    CCoV = 0  # Common noise CoV (%)
    ICoV = 0  # Independent noise CoV (%)
    signal_type = "sinusoid.hz"  # Options:  "sinusoid.hz" -- "trapezoid" -- "triangular" -- "step"
    freq = 2  # Frequency for sinusoid

    ##-------------------------------------------------- ##

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

    all_neurons = NeuronFactory.create_neuron_pool(number_of_neurons=neuron_pool_size)
    neurons_to_simulate = [all_neurons[i] for i in neuron_indexes]

    # TODO Run simulation for selected neurons with iunput CI (possible make a new method in SIMULATION class to run many simulations?)

    # Create a list of (neuron-object , (v, sp) ) data-pairs
    simulation_results = [
        (
            b,
            LIF_Model1.simulate_neuron(
                T, DT, neuron=b, Iinj=CI[: int(T / DT), b.number]
            ),
        )
        for b in neurons_to_simulate
    ]

    # TODO Some Plotting choices

    Output_plots(CI, simulation_results)
    plt.show()


if __name__ == "__main__":
    _main()
