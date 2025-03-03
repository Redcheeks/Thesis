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

    ## PLOT I_rheobase vs R
    ax1.plot(
        [item.R_Mohm for item in neuron_pool_list],
        [item.I_rheobase for item in neuron_pool_list],
        "b",
    )
    ax1.set_xlabel("R [MΩ]")
    ax1.set_ylabel("I_th (Rheobase) [nA]")
    ax1.set_title("I_th - R from Soma_size ")

    ## PLOT I_rheobase distribution vs R
    ax2.plot(
        [item.R_Mohm for item in neuron_pool_list],
        [item.I_rheo_distr for item in neuron_pool_list],
        "b",
    )
    ax2.set_xlabel("R [MΩ]")
    ax2.set_ylabel("I_th  [nA]")
    ax2.set_title("Rheo from distr. - R from Soma_size ")

    ## PLOT I_rheobase distribution vs R from rheobase
    ax3.plot(
        [item.R_I_Mohm for item in neuron_pool_list],
        [item.I_rheo_distr for item in neuron_pool_list],
        "b",
    )
    ax3.set_xlabel("R [MΩ]")
    ax3.set_ylabel("I_th  [nA]")
    ax3.set_title("Rheo from distr. - R from Rheo-distr. ")

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
    plt.show()


if __name__ == "__main__":
    _main()
