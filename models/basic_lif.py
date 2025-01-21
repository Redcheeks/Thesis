import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def run_LIF(pars, Iinj, stop=False):
    """
    Simulate the LIF dynamics with external input current

    Args:
      pars       : parameter dictionary
      Iinj       : input current [pA]. The injected current here can be a value
                   or an array
      stop       : boolean. If True, use a current pulse

    Returns:
      rec_v      : membrane potential
      rec_sp     : spike times
    """

    # Set parameters
    V_th, V_reset = pars["V_th"], pars["V_reset"]
    tau_m, g_L = pars["tau_m"], pars["g_L"]
    V_init, E_L = pars["V_init"], pars["E_L"]
    dt, range_t = pars["dt"], pars["range_t"]
    Lt = range_t.size
    tref = pars["tref"]

    # Initialize voltage
    v = np.zeros(Lt)
    v[0] = V_init

    # Set current time course
    Iinj = Iinj * np.ones(Lt)

    # If current pulse, set beginning and end to 0
    if stop:
        Iinj[: int(len(Iinj) / 2) - 1000] = 0
        Iinj[int(len(Iinj) / 2) + 1000 :] = 0

    # Loop over time
    rec_spikes = []  # record spike times
    tr = 0.0  # the count for refractory duration

    for it in range(Lt - 1):

        if tr > 0:  # check if in refractory period
            v[it] = V_reset  # set voltage to reset
            tr = tr - 1  # reduce running counter of refractory period

        elif v[it] >= V_th:  # if voltage over threshold
            rec_spikes.append(it)  # record spike event
            v[it] = V_reset  # reset voltage
            tr = tref / dt  # set refractory time

        # Calculate the increment of the membrane potential
        dv = (-(v[it] - E_L) + (Iinj[it] / g_L)) * (dt / tau_m)

        # Update the membrane potential
        v[it + 1] = v[it] + dv

    # Get spike times in ms
    rec_spikes = np.array(rec_spikes) * dt

    return v, rec_spikes


def default_pars(**kwargs):  # FROM NEURONMATCH
    pars = {}

    # typical neuron parameters#
    pars["V_th"] = -55.0  # spike threshold [mV]
    pars["V_reset"] = -75.0  # reset potential [mV]
    pars["tau_m"] = 10.0  # membrane time constant [ms]
    pars["g_L"] = 10.0  # leak conductance [nS]
    pars["V_init"] = -75.0  # initial potential [mV]
    pars["E_L"] = -75.0  # leak reversal potential [mV]
    pars["tref"] = 2.0  # refractory time (ms)

    # simulation parameters #
    pars["T"] = 400.0  # Total duration of simulation [ms]
    pars["dt"] = 0.1  # Simulation time step [ms]

    # external parameters if any #
    for k in kwargs:
        pars[k] = kwargs[k]

    pars["range_t"] = np.arange(
        0, pars["T"], pars["dt"]
    )  # Vector of discretized time points [ms]

    return pars


def caillet_random_pars(num_neurons=300):  # Randomly sample parameters for 300 neurons
    # Define the mean and standard deviation for the soma diameter (FIND REAL VALUES!!)
    soma_diameter_mean = 50  # in micrometers
    soma_diameter_std = 5  # in micrometers

    # Generate random soma diameters for each neuron
    soma_diameters = np.random.normal(
        soma_diameter_mean, soma_diameter_std, num_neurons
    )

    # Simulation parameters (same for all neurons)
    T = 400.0  # Total duration of simulation [ms]
    dt = 0.1  # Simulation time step [ms]

    # Define the refractory period (assumed to be 2 ms for all neurons)
    refractory_time_ms = 2  # in ms

    # Create a list to store parameters for each neuron
    neuron_list = []

    for i in range(num_neurons):
        D_soma = soma_diameters[i]  # Soma diameter in micrometers

        # Calculate dependent parameters using empirical relationships
        R = 9.6e5 * (D_soma**-2.4)  # Input resistance [MΩ]
        C = 1.2 * D_soma  # Membrane capacitance [nF]
        tau = 2.6e4 * (D_soma**-1.5)  # Membrane time constant [ms]
        I_th = 9.0e-4 * (D_soma**2.5)  # Rheobase current [nA]
        AHP = 2.5e4 * (D_soma**-1.5)  # Afterhyperpolarization duration [ms]
        ACV = 4.0 * (D_soma**0.7)  # Axonal conduction velocity [m/s]

        # Calculate leak conductance (g_L = C / tau)
        g_L = C / tau  # in μS (since C is in nF and tau is in ms)

        # Set other parameters
        V_rest = -65  # Resting potential in mV
        V_th = -50  # Threshold potential in mV
        V_reset = -70  # Reset potential in mV
        V_init = V_rest  # Initial potential in mV

        # Store parameters for the neuron
        neuron_list.append(
            {
                "soma_diameter": D_soma,
                "R": R,  # Input resistance [MΩ]
                "C": C,  # Membrane capacitance [nF]
                "tau": tau,  # Membrane time constant [ms]
                "I_th": I_th,  # Rheobase current [nA]
                "AHP": AHP,  # Afterhyperpolarization duration [ms]
                "ACV": ACV,  # Axonal conduction velocity [m/s]
                "g_L": g_L,  # Leak conductance [μS]
                "V_rest": V_rest,  # Resting potential [mV]
                "V_th": V_th,  # Threshold potential [mV]
                "V_reset": V_reset,  # Reset potential [mV]
                "V_init": V_init,  # Initial potential [mV]
                "refractory_time": refractory_time_ms,  # Refractory period [ms]
                "T": T,  # Total duration of simulation [ms]
                "dt": dt,  # Simulation time step [ms]
                "range_t": np.arange(
                    0, T, dt
                ),  # Vector of discretized time points [ms]
            }
        )

    # Sort the neurons by soma diameter in ascending order
    neuron_list.sort(key=lambda x: x["soma_diameter"])

    # Convert the sorted list to a dictionary for compatibility
    pars = {i: neuron_list[i] for i in range(num_neurons)}

    return pars


def caillet_quadratic(
    num_neurons=300,
):  # Sample parameters for 300 neurons using quadratic formula

    # Generate normalized indices
    i_values = np.linspace(0, 1, num_neurons)

    soma_Dmin = 50  # minimum soma diameter in micro meter
    soma_Dmax = 100  # maximum soma diameter in micro meter

    # Calculate soma diameters using a quadratic relationship
    soma_diameters = soma_Dmin + (i_values**2) * (soma_Dmax - soma_Dmin)

    # Simulation parameters (same for all neurons)
    T = 400.0  # Total duration of simulation [ms]
    dt = 0.1  # Simulation time step [ms]

    # Define the refractory period (assumed to be 2 ms for all neurons)
    refractory_time_ms = 2  # in ms

    # Create a list to store parameters for each neuron
    neuron_list = []

    for i in range(num_neurons):
        D_soma = soma_diameters[i]  # Soma diameter in micrometers

        # Calculate dependent parameters using empirical relationships
        I_th = 3.85e-9 * 9.1 ** ((i / num_neurons) ** 1.831)  # Rheobase current [A]
        S = 3.96e-4 * I_th**0.396  # (Neuron surface area) in square meters [m^2]
        R = 1.68e-10 * S**-2.43  # Input resistance [Ω]
        C = 7.9e-5 * D_soma  # Membrane capacitance [F]
        tau = 7.9e-5 * D_soma * R  # Membrane time constant [s]
        t_ref = 0.2 * 2.7e-8 * D_soma**-1.51  # (Refractory time) in [s]

        # adjust units
        I_th = I_th * 1e9  # Rheobase current [nA]
        R = R * 1e-6  # Input resistance [MΩ]
        C = C * 1e9  # Membrane capacitance [nF]
        tau = tau * 1e3  # Membrane time constant [ms]
        t_ref = t_ref * 1e3  # (Refractory time) in [ms]

        # Calculate leak conductance (g_L = C / tau)
        g_L = C / tau  # in μS (since C is in nF and tau is in ms)

        # Set other parameters
        V_rest = -65  # Resting potential in mV
        V_th = -50  # Threshold potential in mV
        V_reset = -70  # Reset potential in mV
        V_init = V_rest  # Initial potential in mV

        # Store parameters for the neuron
        neuron_list.append(
            {
                "D_soma": D_soma,
                "R": R,  # Input resistance [MΩ]
                "C": C,  # Membrane capacitance [nF]
                "tau": tau,  # Membrane time constant [ms]
                "I_th": I_th,  # Rheobase current [nA]
                "g_L": g_L,  # Leak conductance [μS]
                "V_rest": V_rest,  # Resting potential [mV]
                "V_th": V_th,  # Threshold potential [mV]
                "V_reset": V_reset,  # Reset potential [mV]
                "V_init": V_init,  # Initial potential [mV]
                "t_ref": t_ref,  # Refractory period [ms]
                "T": T,  # Total duration of simulation [ms]
                "dt": dt,  # Simulation time step [ms]
                "range_t": np.arange(
                    0, T, dt
                ),  # Vector of discretized time points [ms]
            }
        )

    # Convert the sorted list to a dictionary for compatibility
    pars = {i: neuron_list[i] for i in range(num_neurons)}

    return pars


# Code for plotting
def plot_GWN(pars, I_GWN):
    """
    Args:
      pars  : parameter dictionary
      I_GWN : Gaussian white noise input

    Returns:
      figure of the gaussian white noise input
    """

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(pars["range_t"][::3], I_GWN[::3], "b")
    plt.xlabel("Time (ms)")
    plt.ylabel(r"$I_{GWN}$ (pA)")
    plt.subplot(122)
    plot_volt_trace(pars, v, sp)
    plt.tight_layout()
    plt.show()


def my_hists(isi1, isi2, cv1, cv2, sigma1, sigma2):
    """
    Args:
      isi1 : vector with inter-spike intervals
      isi2 : vector with inter-spike intervals
      cv1  : coefficient of variation for isi1
      cv2  : coefficient of variation for isi2

    Returns:
      figure with two histograms, isi1, isi2

    """
    plt.figure(figsize=(11, 4))
    my_bins = np.linspace(10, 30, 20)
    plt.subplot(121)
    plt.hist(isi1, bins=my_bins, color="b", alpha=0.5)
    plt.xlabel("ISI (ms)")
    plt.ylabel("count")
    plt.title(r"$\sigma_{GWN}=$%.1f, CV$_{\mathrm{isi}}$=%.3f" % (sigma1, cv1))

    plt.subplot(122)
    plt.hist(isi2, bins=my_bins, color="b", alpha=0.5)
    plt.xlabel("ISI (ms)")
    plt.ylabel("count")
    plt.title(r"$\sigma_{GWN}=$%.1f, CV$_{\mathrm{isi}}$=%.3f" % (sigma2, cv2))
    plt.tight_layout()
    plt.show()


def diff_DC(pars, I_dc=200.0, tau_m=10.0):
    # Run the LIF model to get initial voltage and spikes
    pars["tau_m"] = tau_m

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
        ax=ax_Idc, label="I_dc", valmin=0, valmax=300, valinit=I_dc, valstep=10
    )
    # Create a second horizontal slider to control tau_m
    ax_tau = fig.add_axes([0.25, 0.05, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    tau_slider = Slider(
        ax=ax_tau, label="tau_m ", valmin=2, valmax=20, valinit=tau_m, valstep=2
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


def _main():

    pars = default_pars(T=500)  # Get parameters
    diff_DC(pars)
    plt.show()


# I_dc=(0, 300, 50), tau_m=(2, 20, 10)


if __name__ == "__main__":
    _main()
