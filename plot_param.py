from dataclasses import asdict
import matplotlib.pyplot as plt
import numpy as np
from neuron import NeuronFactory, Neuron

# from models_legacy.mn_creation import caillet_quadratic

# Global variable
# T = 1000  # Simulation Time [ms]
# DT = 0.1  # Time step in [ms]
NUM_NEURONS = 300  # Number of Neurons simulated


def _main():

    # neuron_pool_list = caillet_quadratic(T, DT, NUM_NEURONS)  # Get parameters
    neuron_pool_list = NeuronFactory.create_neuron_pool(NUM_NEURONS)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    ## PLOT I_rheobase vs D_soma
    ax1.plot(
        [item.D_soma for item in neuron_pool_list],
        [item.I_rheobase for item in neuron_pool_list],
        "b",
    )
    ax1.set_xlabel("D_soma [MΩ]")
    ax1.set_ylabel("I_th (Rheobase) [nA]")
    ax1.set_title("I_th - Soma_size ")

    ## PLOT D_soma and tref
    ax2.plot(
        [item.D_soma for item in neuron_pool_list],
        [item.tref for item in neuron_pool_list],
        "b",
    )
    ax2.set_xlabel("D_soma []")
    ax2.set_ylabel("tref  [ms]")
    ax2.set_title("soma size and absolute refractory period")

    ## PLOT D_soma and tref
    ax3.plot(
        [item.D_soma for item in neuron_pool_list],
        [item.AHP_seconds * 1e3 for item in neuron_pool_list],
        "b",
    )
    ax3.set_xlabel("D_soma []")
    ax3.set_ylabel("AHP  [ms]")
    ax3.set_title("soma size and AHP duration")

    # ## PLOT D_soma and
    # ax3.plot(
    #     [item.D_soma for item in neuron_pool_list],
    #     [item.tref for item in neuron_pool_list],
    #     "b",
    # )
    # ax3.set_xlabel("D_soma []")
    # ax3.set_ylabel("tref  [ms]")
    # ax3.set_title("soma size and absolute refractory period")

    ## PLOT D_soma vs R
    ax4.plot(
        [item.R_Mohm for item in neuron_pool_list],
        [item.D_soma for item in neuron_pool_list],
        "k",
    )
    ax4.set_xlabel("R [MΩ]")
    ax4.set_ylabel("D_soma [(μm)]")
    ax4.set_title("D_soma - R from Soma_size ")

    plt.subplots_adjust(hspace=0.5)
    # plt.show()

    # Soma size of mice? Caillet Elife figure 7 shows that R is between 1-8 MOhm

    fig, ax = plt.subplots()
    R_mouse = np.linspace(1 * 1e-9, 8 * 1e-9, 200)  # R in ohm
    D_mouse = 1.7e-2 * np.pow(R_mouse, -0.41)  # D in meters
    S_mouse = D_mouse * 5.5e-3  # 9.5e-5 * np.pow(R_mouse, -0.41)  # S in meters^2

    ax.plot(R_mouse * 1e9, D_mouse)
    ax.set_xlabel("R mouse [MΩ]")
    ax.set_ylabel("D_soma mouse [(μm)]")
    ax.set_title("D_soma - R for MOUSE ")
    plt.show()


if __name__ == "__main__":
    _main()
