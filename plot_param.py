from dataclasses import asdict
import matplotlib.pyplot as plt
import numpy as np
from neuron import NeuronFactory, Neuron
import matplotlib.pylab as pylab

params = {
    "legend.fontsize": "medium",
    "axes.labelsize": "large",
    "axes.titlesize": "large",
    "xtick.labelsize": "medium",
    "ytick.labelsize": "medium",
}
pylab.rcParams.update(params)
params = {"mathtext.default": "regular"}
plt.rcParams.update(params)


# Global variable
NUM_NEURONS = 300  # Number of Neurons in the pool


def figure_9(neuron_pool):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 5))

    ## PLOT calculated I_rheobase vs D_soma
    ax1.plot(
        [item.D_soma for item in neuron_pool],
        [7.8e2 * np.pow(item.D_soma_meter, 2.52) * 1e9 for item in neuron_pool],
        "k",
    )
    ax1.grid(linestyle="--", linewidth=0.5, axis="y")
    ax1.set_xlabel("$D_{soma}$ [μm]")
    ax1.set_ylabel("Rheobase [nA]")
    ax1.set_title("Calculated Rheobase - Neuron-parameters")

    ## PLOT distribution I_rheo vs D_soma
    ax2.plot(
        [item.D_soma for item in neuron_pool],
        [item.I_rheobase for item in neuron_pool],
        "k",
    )
    ax2.grid(linestyle="--", linewidth=0.5, axis="y")
    ax2.set_xlabel("$D_{soma}$ [μm]")
    ax2.set_ylabel("Rheobase [nA]")
    ax2.set_title("Rheobase - Experimental Distribution")

    ## PLOT D_soma and tref based on Hug et. al. (2023)
    ax3.plot(
        [item.D_soma for item in neuron_pool],
        [item.tref / 0.05 * 0.2 for item in neuron_pool],
        "k",
    )
    ax3.grid(linestyle="--", linewidth=0.5, axis="y")
    ax3.set_xlabel("$D_{soma}$ [μm]")
    ax3.set_ylabel("ARP [ms]")
    ax3.set_title("ARP using ARP = 0.2*AHP")

    ## PLOT D_soma vs tref
    ax4.plot(
        [item.D_soma for item in neuron_pool],
        [item.tref for item in neuron_pool],
        "k",
    )
    ax4.grid(linestyle="--", linewidth=0.5, axis="y")
    ax4.set_xlabel("$D_{soma}$ [μm]")
    ax4.set_ylabel("ARP [ms]")
    ax4.set_title("ARP using ARP = 0.05*AHP")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.14)


def other_params(neuron_pool):

    # Soma size of mice? Caillet Elife figure 7 shows that R is between 1-8 MOhm

    plt.subplots(1, 2, figsize=(10, 5))

    R_mouse = np.linspace(1 * 1e-9, 8 * 1e-9, 200)  # R in ohm
    D_mouse = 1.7e-2 * np.pow(R_mouse, -0.41)  # D in meters
    S_mouse = D_mouse * 5.5e-3  # 9.5e-5 * np.pow(R_mouse, -0.41)  # S in meters^2

    plt.subplot(1, 2, 1)
    plt.plot(R_mouse * 1e9, D_mouse, "k")
    plt.grid(linestyle="--", linewidth=0.5, axis="y")
    plt.xlabel("Resistance $R$ [MΩ]")
    plt.ylabel("$D_{soma}$ [μm]")
    plt.title("Caillet Data for Mouse Soma Size ")

    plt.subplot(1, 2, 2)
    plt.plot(
        [item.R_Mohm for item in neuron_pool],
        [item.D_soma for item in neuron_pool],
        "k",
    )
    plt.grid(linestyle="--", linewidth=0.5, axis="y")
    plt.xlabel("Resistance $R$ [MΩ]")
    plt.ylabel("$D_{soma}$ [μm]")
    plt.title("Scaled Human-Model Soma Size ")
    plt.tight_layout()

    # _______________________________#

    plt.subplots(2, 2, figsize=(10, 5))

    plt.subplot(2, 2, 1)
    plt.plot(
        [item.D_soma for item in neuron_pool],
        [item.R_Mohm for item in neuron_pool],
        "k",
    )
    plt.grid(linestyle="--", linewidth=0.5, axis="y")
    plt.xlabel("$D_{soma}$ [μm]")
    plt.ylabel("Resistance $R$ [MΩ]")
    plt.title("Neuron Membrane Resistance", fontsize="large")
    # _____
    plt.subplot(2, 2, 2)
    plt.plot(
        [item.D_soma for item in neuron_pool],
        [item.AHP_seconds * 1e3 for item in neuron_pool],
        "k",
    )
    plt.grid(linestyle="--", linewidth=0.5, axis="y")
    plt.xlabel("$D_{soma}$ [μm]")
    plt.ylabel("AHP [ms]")
    plt.title("After-Hyperpolarization-Period", fontsize="large")
    # _____
    plt.subplot(2, 2, 3)
    plt.plot(
        [item.D_soma for item in neuron_pool],
        [item.tref for item in neuron_pool],
        "k",
    )
    plt.grid(linestyle="--", linewidth=0.5, axis="y")
    plt.xlabel("$D_{soma}$ [μm]")
    plt.ylabel("ARP [ms]")
    plt.title("After-Repolarization-Period (Delayed Depolarization)", fontsize="large")
    # _____
    plt.subplot(2, 2, 4)
    plt.plot(
        [item.number for item in neuron_pool],
        [item.I_rheobase for item in neuron_pool],
        "k",
    )
    plt.grid(linestyle="--", linewidth=0.5, axis="y")
    plt.xlabel("$D_{soma}$ [μm]")
    plt.ylabel("Rheobase [nA]")
    plt.title("Neuron Rheobase Current", fontsize="large")
    # _______________________________#

    plt.tight_layout()


def rheo_threshold(neuron_pool):
    plt.subplots(1, 2, figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(
        [item.D_soma for item in neuron_pool],
        [item.Rheobase_threshold for item in neuron_pool],
        "k",
    )
    plt.ylabel("Rheobase Threshold [nA]")
    plt.xlabel("$D_{soma}$ [μm]")
    plt.title("Rheobase Threshold using coefficients {1.05 - 1.62}")
    plt.grid(linestyle="--", linewidth=0.5, axis="y")

    plt.subplot(1, 2, 2)
    plt.plot(
        [item.D_soma for item in neuron_pool],
        [item.Rheobase_threshold * 2 for item in neuron_pool],
        "k",
    )
    plt.ylabel("Rheobase Threshold [nA]")
    plt.xlabel("$D_{soma}$ [μm]")
    plt.title("Rheobase Threshold using coefficients {2.1 - 3.24}")
    plt.grid(linestyle="--", linewidth=0.5, axis="y")

    plt.tight_layout()


def _main():

    neuron_pool = NeuronFactory.create_neuron_pool(
        distribution=True, number_of_neurons=NUM_NEURONS
    )

    figure_9(neuron_pool)

    other_params(neuron_pool)

    rheo_threshold(neuron_pool)

    plt.show()


if __name__ == "__main__":
    _main()
