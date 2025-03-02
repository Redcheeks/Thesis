import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# from simple_inputs import trapez_current, linear_spiking_current
from descending_drive import cortical_input
from models_legacy.LIF_model1 import run_LIF as run_model1
from models_legacy.LIF_model2 import run_LIF as run_model2
from models_legacy.mn_creation import caillet_quadratic

# Global variable
T = 1000  # Simulation Time [ms]
DT = 0.1  # Time step in [ms]
NUM_NEURONS = 300  # Number of Neurons simulated


# calculates and plots F-I curve for a neuron with properties *pars* for current between Imin and Imax
def F_I_plot(
    pars_dict,
    Imin=1,
    Imax=50,
    n_samples=50,
    run_LIF=run_model1,
    neurons=[5, 50, 150, 200, 275],
):

    I_range = np.linspace(start=Imin, stop=Imax, num=n_samples)
    # print(I_range)
    freq = []  # record freq for each current level
    fig, ax = plt.subplots()
    for neuron in neurons:
        for i in I_range:

            v, sp = run_LIF(pars_dict[neuron], Iinj=i, stop=True)

            if sp.size > 1:
                isi = np.diff(sp * 1e-3)  # Compute interspike intervals, time in ms
                # Assign frequency values to corresponding spike times (excluding the first spike)
                freq.append(np.mean((1 / isi[1:])))  # First spike has no frequency

            else:
                freq.append(0)
        line = ax.plot(I_range, freq)
        freq = []

    ax.set_xlabel("Current (nA)")
    ax.set_ylabel("Frequency (HZ)")
    # ax.set_ylim([-80, -40])
    ax.set_title("Frequency-Current Plot")
    ax.legend(neurons)


def Freq_inst_plot(CI, pars_dict, run_LIF=run_model1, neurons=[5, 50, 150, 200, 275]):

    time = np.linspace(0, T, int(T / DT))
    freq = {}  # record freq over time
    fig, ax = plt.subplots()

    for i in neurons:  # for each of the neurons
        v, sp = run_LIF(pars_dict[i], CI[: int(T / DT), i])

        if len(sp) < 2:
            freq = 0  # Not enough spikes to compute frequency
        else:
            isi = np.diff(sp * 1e-3)  # Compute interspike intervals, time in ms

            # Assign frequency values to corresponding spike times (excluding the first spike)
            freq[i] = np.concatenate(([0], 1 / isi))  # First spike has no frequency

            ax.plot(sp, freq[i], ".")

    ax.set_ylabel("Frequency (HZ)")
    ax.set_xlabel("Time (ms)")
    ax.set_title("Frequency-time Plot")
    ax.legend(neurons)


def Output_plot(CI, pars_dict, run_LIF=run_model1, neurons=[5, 50, 150, 200, 275]):

    time = np.linspace(0, T, int(T / DT))

    fig, ax = plt.subplots()
    for i in neurons:
        v, sp = run_LIF(pars_dict[i], CI[: int(T / DT), i])
        if sp.size:
            sp_num = (sp / DT).astype(int) - 1
            v[sp_num] += 20  # draw nicer spikes

        ax.plot(time, v, "-")

    ax.set_ylim([-80, -20])
    ax.axhline(pars_dict[1]["V_th"], color="k", ls="--")
    ax.legend(neurons)


def _main():

    pars_dict = caillet_quadratic(T, DT, NUM_NEURONS)  # Get parameters

    T_dur = T  # Total time in ms
    dt = DT  # Time step in ms
    n_mn = NUM_NEURONS  # Number of motor neurons
    n_clust = 5  # Number of clusters
    max_I = 40  # Max input current (nA)
    CCoV = 0  # Common noise CoV (%)
    ICoV = 0  # Independent noise CoV (%)

    ## RUN PARAMETERS ##

    run_model = run_model1  ## SELECT THE MODEL TO RUN!!

    neurons = [1, 50, 150, 200, 250]  ## SELECT WHICH NEURONS TO RUN & PLOT

    CI = cortical_input(n_mn, n_clust, max_I, T_dur, dt, CCoV, ICoV, "sinusoid.hz", 2)

    # time = np.linspace(0, T_dur, int(T / DT))
    Output_plot(CI, pars_dict, run_model, neurons)
    Freq_inst_plot(CI, pars_dict, run_model, neurons)
    # plt.plot(CI[: int(T / DT), neurons])
    # Output_plot(CI, pars_dict, run_model, neurons)
    # F_I_plot(pars_dict, Imin=1, Imax=50, n_samples=50, neurons=neurons)
    plt.show()


if __name__ == "__main__":
    _main()


## Can edit settings.json to make this one always be the one running with:

# "python.terminal.launchArgs": [    "${workspaceFolder}/run_MAIN.py"],
