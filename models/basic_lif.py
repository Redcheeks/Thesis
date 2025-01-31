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
    R_m = pars["R"]
    dt, range_t = pars["dt"], pars["range_t"]
    Lt = range_t.size
    tref = 2e-3  # pars["tref"]  # here is the issue!! t_ref should be around 2

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
        dv = (-(v[it] - E_L) + (Iinj[it] * R_m)) * (dt / tau_m)

        # Update the membrane potential
        v[it + 1] = v[it] + dv

    # Get spike times in ms
    rec_spikes = np.array(rec_spikes) * dt
    # print(rec_spikes)

    return v, rec_spikes


def default_pars(**kwargs):  # FROM NEURONMATCH
    pars = {}

    # typical neuron parameters#
    pars["V_th"] = -55.0  # spike threshold [mV]
    pars["V_reset"] = -75.0  # reset potential [mV]
    pars["tau_m"] = 20.0  # membrane time constant [ms]
    pars["g_L"] = 10.0  # leak conductance [nS]
    pars["V_init"] = -75.0  # initial potential [mV]
    pars["E_L"] = -75.0  # leak reversal potential [mV]
    pars["tref"] = 2.0  # refractory time (ms)
    pars["R"] = 1 / pars["g_L"]

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


# Sample parameters for 300 neurons using quadratic formula
def caillet_quadratic(num_neurons=300):

    # Generate normalized indices
    i_values = np.linspace(0, 1, num_neurons)

    soma_Dmin = 50e-6  # minimum soma diameter in meter
    soma_Dmax = 100e-6  # maximum soma diameter in meter

    # Calculate soma diameters using a quadratic relationship
    soma_diameters = soma_Dmin + (i_values**2) * (soma_Dmax - soma_Dmin)

    # Simulation parameters (same for all neurons)
    T = 0.6  # Total duration of simulation [s]
    dt = 0.1e-3  # Simulation time step [s]

    # Define the refractory period (assumed to be 2 ms for all neurons)
    t_ref_unit = 2e-3  # in s

    # Create a list to store parameters for each neuron
    neuron_list = []

    for i in range(num_neurons):
        D_soma_unit = soma_diameters[i]  # Soma diameter in meters

        # Calculate dependent parameters using empirical relationships
        # I_th_unit = 3.85e-9 * np-pow(9.1, ((i / num_neurons) ** 1.831))
        # S_unit = 3.96e-4 * np.pow(I_th_unit, 0.396)

        S_unit = 5.5e-3 * D_soma_unit  # (Neuron surface area) in square meters [m^2]
        I_th_unit = 7.8e2 * np.pow(D_soma_unit, 2.52)  # Rheobase current [A]
        R_unit = 1.7e-10 * np.pow(S_unit, -2.43)  # Input resistance [Ω]
        C_unit = 7.9e-5 * D_soma_unit  # Membrane capacitance [F]
        tau_unit = 7.9e-5 * D_soma_unit * R_unit  # Membrane time constant [s]
        # t_ref_unit = (0.2 * 2.7e-8 * np.pow(D_soma_unit, -1.51))  # (Refractory time) in [s] based on: AHP -

        # adjust units
        # I_th = I_th_unit * 1e9  # Rheobase current [nA]
        # R = R_unit * 1e-6  # Input resistance [MΩ]
        # C = C_unit * 1e9  # Membrane capacitance [nF]
        # tau = tau_unit * 1e3  # Membrane time constant [ms]
        # t_ref = t_ref_unit * 1e3  # (Refractory time) in [ms]
        # D_soma = D_soma_unit * 1e6  # Soma diameter in [μm]

        # Calculate leak conductance (g_L = C / tau)
        g_L_unit = C_unit / tau_unit  # in μS (since C is in nF and tau is in ms)

        # Set other parameters
        E_L_unit = -75e-3  # leak reversal potential in mV # Resting potential in mV
        V_th_unit = -50e-3  # Threshold potential in mV
        V_reset_unit = -70e-3  # Reset potential in mV
        V_init_unit = E_L_unit  # Initial potential in mV

        # Store parameters for the neuron
        neuron_list.append(
            {
                "number": i,
                "D_soma": D_soma_unit,
                "R": R_unit,  # Input resistance [MΩ]
                "C": C_unit,  # Membrane capacitance [nF]
                "tau_m": tau_unit,  # Membrane time constant [ms]
                "I_th": I_th_unit,  # Rheobase current [nA]
                "g_L": g_L_unit,  # Leak conductance [μS]
                "V_rest": E_L_unit,  # Resting potential [mV]
                "V_th": V_th_unit,  # Threshold potential [mV]
                "V_reset": V_reset_unit,  # Reset potential [mV]
                "V_init": V_init_unit,  # Initial potential [mV]
                "E_L": E_L_unit,  # leak reversal potential [mV]
                "tref": t_ref_unit,  # Refractory period [ms]
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


def diff_DC(pars, I_dc=10e-12, tau_m=10e-3):  # Plot interactively I_DC
    # Run the LIF model to get initial voltage and spikes
    # pars["tau_m"] = tau_m

    v, sp = run_LIF(pars, Iinj=I_dc, stop=True)
    V_th = pars["V_th"]
    dt, range_t = pars["dt"], pars["range_t"]
    if sp.size:
        sp_num = (sp / dt).astype(int) - 1
        v[sp_num] += 20e-3  # draw nicer spikes

    # Plot the initial voltage trace
    fig, ax = plt.subplots()
    (line,) = ax.plot(pars["range_t"], v, "b")
    ax.axhline(pars["V_th"], color="k", ls="--")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("V (mV)")
    ax.legend(["Membrane\npotential", r"Threshold V$_{\mathrm{th}}$"])
    # ax.set_ylim([-80, -40])
    ax.set_title("Voltage Response for DC current")

    # Adjust the main plot to make room for the slider
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # Create a horizontal slider to control I_dc
    ax_Idc = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    Idc_slider = Slider(
        ax=ax_Idc, label="I_dc", valmin=0, valmax=50e-12, valinit=I_dc, valstep=1e-12
    )
    # Create a second horizontal slider to control tau_m
    ax_tau = fig.add_axes([0.25, 0.05, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    tau_slider = Slider(
        ax=ax_tau,
        label="tau_m ",
        valmin=2e-3,
        valmax=20e-3,
        valinit=pars["tau_m"],
        valstep=2e-3,
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


def F_I_SingleNeuron(pars, Imin=1e-12, Imax=50e-12, n_samples=50):
    # calculates and plots F-I curve for a neuron with properties *pars* for current between Imin and Imax
    I_range = np.linspace(start=Imin, stop=Imax, num=n_samples)
    print(I_range)
    freq = []  # record freq for each current level

    for i in I_range:

        v, sp = run_LIF(pars, Iinj=i, stop=True)
        if sp.size > 0:
            freq.append(1 / (sp[1] - sp[0]))
        else:
            freq.append(0)

    fig, ax = plt.subplots()
    line = ax.plot(I_range, freq, "b")
    ax.set_xlabel("Current (A)")
    ax.set_ylabel("Frequency (HZ)")

    # ax.set_ylim([-80, -40])
    ax.set_title("Frequency-Current Plot")


def F_I_MultiNeuron(pars_list, Imin=1e-12, Imax=50e-12, n_samples=50):
    # calculates and plots F-I curve for a neuron with properties *pars* for current between Imin and Imax
    I_range = np.linspace(start=Imin, stop=Imax, num=n_samples)

    freq = []  # record freq for each current level
    fig, ax = plt.subplots()
    it = 0
    for pars in pars_list:  # for each of the neurons
        it += 1
        for I_test in I_range:

            v, sp = run_LIF(pars, Iinj=I_test, stop=False)
            if sp.size > 1:
                freq.append(1 / (sp[1] - sp[0]))
            else:
                freq.append(0)
        ax.plot(I_range, freq, "x-")
        freq = []  # reset freq

    ax.set_xlabel("Current (nA)")
    ax.set_ylabel("Frequency (HZ)")

    # ax.set_ylim([0, 60])
    ax.set_title("Frequency-Current Plot")


def _main():
    pars = caillet_quadratic()  # Get parameters
    # print(pars[250]["tau_m"], pars[150]["tau_m"])
    diff_DC(pars[100], 19e-12)
    # diff_DC(pars[200], 19)
    # F_I_SingleNeuron(pars[50])
    # F_I_MultiNeuron([pars[10], pars[50], pars[150], pars[200], pars[250]], Imax=50e-12)
    # plt.legend(["10", "50", "150", "200", "250"])
    plt.show()


if __name__ == "__main__":
    _main()
