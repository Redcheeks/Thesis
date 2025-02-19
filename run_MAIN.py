import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# from simple_inputs import trapez_current, linear_spiking_current
from descending_drive import cortical_input
from models.LIF_model1 import run_LIF as run_model1
from models.LIF_model2 import run_LIF as run_model2
from models.mn_creation import caillet_quadratic

# Global variable
T = 1000  # Simulation Time [ms]
DT = 0.1  # Time step in [ms]
NUM_NEURONS = 300  # Number of Neurons simulated


# Plot interactively I_DC
def diff_DC(pars, I_dc=10.0, tau_m=10.0):
    # Run the LIF model to get initial voltage and spikes
    # pars["tau_m"] = tau_m

    v, sp = run_LIF(pars, Iinj=I_dc, stop=True)
    V_th = pars["V_th"]
    dt, range_t = pars["dt"], pars["range_t"]
    if sp.size:
        sp_num = (sp / dt).astype(int) - 1
        v[sp_num] += 20  # draw nicer spikes

    # Plot the initial voltage trace
    fig, ax = plt.subplots()
    (line,) = ax.plot(pars["range_t"], v, "b")
    ax.axhline(pars["V_th"], color="k", ls="--")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("V (mV)")
    ax.legend(["Membrane\npotential", r"Threshold V$_{\mathrm{th}}$"])
    ax.set_ylim([-80, -40])
    ax.set_title("Voltage Response for DC current")

    # Adjust the main plot to make room for the slider
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # Create a horizontal slider to control I_dc
    ax_Idc = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    Idc_slider = Slider(
        ax=ax_Idc, label="I_dc", valmin=0, valmax=100, valinit=I_dc, valstep=1
    )
    # Create a second horizontal slider to control tau_m
    ax_tau = fig.add_axes([0.25, 0.05, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    tau_slider = Slider(
        ax=ax_tau, label="tau_m ", valmin=2, valmax=20, valinit=pars["tau_m"], valstep=2
    )

    # Update function to be called when the slider's value changes
    def update(val):
        new_I_dc = Idc_slider.val
        pars["tau_m"] = tau_slider.val
        v, sp = run_LIF(pars, Iinj=new_I_dc, stop=True)
        # Update spikes visualization
        if sp.size:
            sp_num = (sp / dt).astype(int) - 1
            v[sp_num] += 20
        line.set_ydata(v)
        fig.canvas.draw_idle()

    # Register the update function with the slider
    Idc_slider.on_changed(update)
    tau_slider.on_changed(update)

    if current_func == trapez_current:

        I_time, I = current_func(T, DT, max_I, 100)

    else:  # elif current_func == linear_spiking_current:
        I_time, I = current_func(T, DT, 20, 100, spike_amp, spike_freq)

    v, sp = run_LIF(pars, Iinj=I, stop=False)
    V_th = pars["V_th"]
    range_t = pars["range_t"]
    if sp.size:
        sp_num = (sp / DT).astype(int) - 1
        v[sp_num] += 20  # draw nicer spikes

    # Plot the initial voltage trace
    fig, ax = plt.subplots()
    (current_line,) = ax.plot(I_time, (I + pars["V_reset"]), "r--")
    (line,) = ax.plot(pars["range_t"], v, "b")
    ax.axhline(pars["V_th"], color="k", ls="--")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("V (mV)")
    ax.legend(["Current Input", "Membrane\npotential", r"Threshold V$_{\mathrm{th}}$"])
    ax.set_ylim([-80, -20])
    ax.set_title("Voltage Response for current I")

    # Adjust the main plot to make room for the slider
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # Create a horizontal slider to control I_dc
    ax_I = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    I_slider = Slider(
        ax=ax_I, label="max_I [nA]", valmin=1, valmax=50, valinit=np.max(I), valstep=1
    )

    # Update function to be called when the slider's value changes
    def update(val):

        if current_func == trapez_current:
            I_time, I = current_func(T, DT, I_slider.val, 100)

        else:  # elif current_func == linear_spiking_current:
            I_time, I = current_func(T, DT, I_slider.val, 100, spike_amp, spike_freq)

        v, sp = run_LIF(pars, Iinj=I, stop=False)
        # Update spikes visualization
        if sp.size:
            sp_num = (sp / DT).astype(int) - 1
            v[sp_num] += 20
        line.set_ydata(v)
        current_line.set_ydata(I + pars["V_reset"])
        fig.canvas.draw_idle()

    # Register the update function with the slider
    I_slider.on_changed(update)


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
                freq.append(1e3 / np.mean(np.diff(sp)))
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
    max_I = 100  # Max input current (nA)
    CCoV = 0  # Common noise CoV (%)
    ICoV = 0  # Independent noise CoV (%)

    ## RUN PARAMETERS ##
    run_model = run_model1
    neurons = [150, 200]
    # pars_dict[5]["doublet_current"]

    CI = cortical_input(n_mn, n_clust, max_I, T_dur, dt, CCoV, ICoV, "sinusoid.hz", 2)
    # time = np.linspace(0, T_dur, int(T / DT))
    # Output_plot(CI, pars_dict, neurons=[5])
    # Freq_plot(CI, pars_dict, neurons=[1, 5, 20, 50, 150, 200, 250])
    plt.plot(CI[: int(T / DT), neurons])
    Output_plot(CI, pars_dict, run_model, neurons)
    # F_I_plot(pars_dict, Imin=1, Imax=50, n_samples=50, neurons=[1, 5, 50, 150, 200, 275])
    plt.show()


if __name__ == "__main__":
    _main()
