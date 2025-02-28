from models.mn_creation import caillet_quadratic
import numpy as np
import matplotlib.pyplot as plt

# Global variable
T = 1000  # Simulation Time [ms]
DT = 0.1  # Time step in [ms]
NUM_NEURONS = 300  # Number of Neurons simulated


def _main():

    pars_dict = caillet_quadratic(T, DT, NUM_NEURONS)  # Get parameters

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    ## PLOT I_rheobase vs R
    for i in pars_dict:
        ax1.plot(pars_dict[i]["R"], pars_dict[i]["I_th"], "k.")
    ax1.set_xlabel("R [MΩ]")
    ax1.set_ylabel("I_th (Rheobase) [nA]")
    # ax.set_ylim([-80, -40])
    ax1.set_title("I_th - R ")

    ## PLOT tau vs R

    for i in pars_dict:
        ax2.plot(pars_dict[i]["R"], pars_dict[i]["tau_m"], "k.")
    ax2.set_xlabel("R [MΩ]")
    ax2.set_ylabel("τ [ms]")
    # ax.set_ylim([-80, -40])
    ax2.set_title("τ - R ")

    ## PLOT S_soma vs R

    for i in pars_dict:
        ax3.plot(pars_dict[i]["R"], pars_dict[i]["S_soma"], "k.")
    ax3.set_xlabel("R [MΩ]")
    ax3.set_ylabel("S_soma [m * 10^(-6)]")
    # ax.set_ylim([-80, -40])
    ax3.set_title("S_soma - R ")

    plt.subplots_adjust(hspace=0.5)
    plt.show()


if __name__ == "__main__":
    _main()
