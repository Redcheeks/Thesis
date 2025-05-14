import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Type
from neuron import Neuron, NeuronFactory
from simulation.models.LIF_model_SIMPLE import LIF_SIMPLE
from simulation.models.LIF_model1 import LIF_Model1
from simulation.models.LIF_model2_v3 import LIF_Model2v3
from simulation.models.LIF_model3_v2 import LIF_Model3v2

## ------------- Simulation Parameters ------------- ##
T = 100  # Simulation Time [ms]
DT = 1  # Time step in [ms]
neuron_pool_size = 300  # Total number of Neurons in the pool

## ------ Cortical input Simulation Parameters ------ ##
min_I = 5
max_I = 50  # Max input current (nA)

## ------ Neurons to be modelled & plotted. ------ ##
NEURON_INDEXES: List[int] = [10, 50, 150, 200]  # choose 4 neurons to plot!


def FI_curves(neuron, model):

    for i, Iinj in enumerate(np.linspace(min_I, max_I, (max_I - min_I) * 2)):
        current = Iinj * np.ones(int(T / DT))
        v, sp = model.simulate_neuron(T, DT, neuron, current)[:2]
        # only interested in 2nd return value
        if len(sp) > 1:
            freq = 1 / np.mean(np.diff(sp)) * 1e3
        else:
            freq = 0

        plt.plot(Iinj, freq, "o")
        plt.xlabel("Current [nA]")
        plt.ylabel("Frequency [Hz]")


if __name__ == "__main__":
    # Create neurons with NEURONFACTORY class (# Generate list of new neurons)
    all_neurons = NeuronFactory.create_neuron_pool(
        distribution=True, number_of_neurons=neuron_pool_size
    )
    neurons = [all_neurons[i] for i in NEURON_INDEXES]

    ## -------- I-F Curve for basic LIF -------- ##

    plt.figure(figsize=(10, 5))
    model = LIF_SIMPLE
    plt.suptitle("F-I curves for Simple LIF model")

    for it, neuron in enumerate(neurons):  # for each neuron

        plt.subplot(2, 2, it + 1)
        FI_curves(neuron, model)
        plt.title(f"Neuron {neuron.number}")
        plt.subplots_adjust(hspace=0.5, wspace=0.5)

    ## -------- I-F Curve for model 1 -------- ##

    plt.figure(figsize=(10, 5))
    model = LIF_Model1
    plt.suptitle("F-I curves for LIF model 1 ")

    for it, neuron in enumerate(neurons):  # for each neuron

        plt.subplot(2, 2, it + 1)
        FI_curves(neuron, model)
        plt.title(f"Neuron {neuron.number}")
        plt.subplots_adjust(hspace=0.5, wspace=0.5)

    ## -------- I-F Curve for model 2 -------- ##

    plt.figure(figsize=(10, 5))
    model = LIF_Model2v3
    plt.suptitle("F-I curves for LIF model 2v3 ")

    for it, neuron in enumerate(neurons):  # for each neuron

        plt.subplot(2, 2, it + 1)
        FI_curves(neuron, model)
        plt.title(f"Neuron {neuron.number}")
        plt.subplots_adjust(hspace=0.5, wspace=0.5)

    ## -------- I-F Curve for model 3 -------- ##

    plt.figure(figsize=(10, 5))
    model = LIF_Model3v2
    plt.suptitle("F-I curves for LIF model 3v2 ")

    for it, neuron in enumerate(neurons):  # for each neuron

        plt.subplot(2, 2, it + 1)
        FI_curves(neuron, model)
        plt.title(f"Neuron {neuron.number}")
        plt.subplots_adjust(hspace=0.5, wspace=0.5)

    plt.show()
