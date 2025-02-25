import numpy as np

##MODEL BASED ON GAIN/EXCITABILITY##


# Global variable
T = 1000  # Simulation Time [ms]
DT = 0.1  # Time step in [ms]
NUM_NEURONS = 300  # Number of Neurons simulated


def run_LIF(pars, Iinj, stop=False):
    """
    Simulate the LIF dynamics with external input current

    Args:
      pars       : parameter dictionary
      Iinj       : input current [nA]. The injected current here can be a value
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
    range_t = pars["range_t"]
    Lt = range_t.size
    tref = pars["tref"]
    gain_leak = pars["gain_leak"]
    gain_exc = pars["gain_exc"]
    d_soma = pars["D_soma"]
    doublet_current = pars["Doublet_current"]
    # print(doublet_current)

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
    last_spike_counter = (
        100 / DT
    )  # time since spike, used for doublet interval (3-10ms).
    renshaw_inhib = False  # is renshaw cell inhibiting
    relax_counter = 200 / DT  # used to check for relaxation period for renshaw state.

    # excitability parameter that effects dv increase.
    max_exc = gain_exc * 2
    excitability = max_exc  # start with high excitability
    depression_reset = (
        100
        / DT  # 100ms for depression to wear off? (this should be based on literature)
    )
    # TODO: spiking should decrease excitability.
    # this effect should wear off over time, a new spike decreases it again though
    # (many spikes quickly following eachother keeps a very decreased excitability)

    doub_count = 0  # For testing purposes

    for it in range(Lt - 1):

        # TODO : this maybe should be separate????
        if relax_counter > 100 / DT:  # Check if in relaxed state for renshaw
            renshaw_inhib = False
            excitability = max_exc

        if tr > 0:  # check if in refractory period
            v[it] = V_reset  # set voltage to reset
            tr = tr - 1  # reduce running counter of refractory period

        elif v[it] >= V_th:  # if voltage over threshold
            rec_spikes.append(it)  # record spike event

            if (
                relax_counter > 100 / DT
            ):  # check if neuron was relaxed prior to this spike
                renshaw_inhib = False
            else:
                renshaw_inhib = True

            if (3 / DT) < last_spike_counter < (10 / DT) and renshaw_inhib == False:
                v[it - 1] = V_th + 20  ##ONLY FOR MAKING DOUBLET SPIKES MORE VISIBLE!!
                tr = tref * 2 / DT  # set new refractory time : double normal time.
                doub_count += 1
                excitability = gain_exc * 0.8  # if this is a doublet, more depression??
            else:
                excitability = gain_exc
                tr = tref / DT  # set refractory time

            last_spike_counter = 0.0
            relax_counter = 0.0
            v[it] = V_reset  # reset voltage

        # Calculate the increment of the membrane potential with variable gains

        dv = (-(gain_leak) * (v[it] - E_L) + (excitability) * (Iinj[it] * R_m)) * (
            DT / tau_m
        )

        # Update the membrane potential [mv]
        v[it + 1] = v[it] + dv
        relax_counter += 1
        last_spike_counter += 1
        # depression wears off
        if excitability < max_exc:
            excitability += (
                DT / depression_reset
            )  # After reset time the excitability should reset

    # Get spike times in ms
    rec_spikes = np.array(rec_spikes) * DT
    print(doub_count)

    return v, rec_spikes
