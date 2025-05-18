from dataclasses import asdict
import matplotlib.pyplot as plt
import numpy as np
from neuron import NeuronFactory, Neuron


# Global variable
# T = 1000  # Simulation Time [ms]
# DT = 0.1  # Time step in [ms]
NUM_NEURONS = 300  # Number of Neurons simulated


def _main():

    neuron_pool = NeuronFactory.create_neuron_pool(
        distribution=True, number_of_neurons=NUM_NEURONS
    )

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 8))

    ## PLOT calculated I_rheobase vs D_soma
    ax1.plot(
        [item.D_soma for item in neuron_pool],
        [7.8e2 * np.pow(item.D_soma_meter, 2.52) * 1e9 for item in neuron_pool],
        "b",
    )
    ax1.set_xlabel("D_soma [μm]")
    ax1.set_ylabel("Rheobase [nA]")
    ax1.set_title("Rheobase extrapolated from neuron-parameters")

    ## PLOT distribution I_rheo vs D_soma
    ax2.plot(
        [item.D_soma for item in neuron_pool],
        [item.I_rheobase for item in neuron_pool],
        "b",
    )
    ax2.set_xlabel("D_soma [μm]")
    ax2.set_ylabel("Rheobase [nA]")
    ax2.set_title("Rheobase from experimental distribution")

    ## PLOT D_soma and tref based on Hug et. al. (2023)
    ax3.plot(
        [item.D_soma for item in neuron_pool],
        [item.tref / 0.05 * 0.2 for item in neuron_pool],
        "k",
    )
    ax3.set_xlabel("D_soma [μm]")
    ax3.set_ylabel("ARP [ms]")
    ax3.set_title("ARP using ARP = 0.2*AHP")

    ## PLOT D_soma vs tref
    ax4.plot(
        [item.D_soma for item in neuron_pool],
        [item.tref for item in neuron_pool],
        "k",
    )
    ax4.set_xlabel("D_soma [μm]")
    ax4.set_ylabel("ARP [ms]")
    ax4.set_title("ARP using ARP = 0.05*AHP")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.14)

    # plt.show()

    # Soma size of mice? Caillet Elife figure 7 shows that R is between 1-8 MOhm

    # fig, ax = plt.subplots()
    # R_mouse = np.linspace(1 * 1e-9, 8 * 1e-9, 200)  # R in ohm
    # D_mouse = 1.7e-2 * np.pow(R_mouse, -0.41)  # D in meters
    # S_mouse = D_mouse * 5.5e-3  # 9.5e-5 * np.pow(R_mouse, -0.41)  # S in meters^2

    # ax.plot(R_mouse * 1e9, D_mouse)
    # ax.set_xlabel("R mouse [MΩ]")
    # ax.set_ylabel("D_soma mouse [(μm)]")
    # ax.set_title("D_soma - R for MOUSE ")

    plt.show()


if __name__ == "__main__":
    _main()
