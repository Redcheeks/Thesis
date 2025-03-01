import numpy as np


T = 1000  # Simulation Time [ms]
DT = 0.1  # Time step in [ms]
NUM_NEURONS = 300  # Number of Neurons simulated


# Sample parameters for 300 neurons using quadratic formula
def caillet_quadratic(T=T, dt=DT, num_neurons=NUM_NEURONS):
    # Simulation parameters (same for all neurons)
    # T- Total duration of simulation [ms]
    # dt - Simulation time step [ms]

    # Generate normalized indices
    i_values = np.linspace(0, num_neurons, num_neurons) / num_neurons

    soma_Dmin = 50e-6  # minimum soma diameter in meter
    soma_Dmax = 100e-6  # maximum soma diameter in meter

    # Calculate soma diameters using a quadratic relationship
    soma_diameters = soma_Dmin + (i_values**2) * (soma_Dmax - soma_Dmin)

    leak_array = np.linspace(0.25, 0.15, NUM_NEURONS)
    doublet_currents_coefficient = np.linspace(
        3.4, 2.1, NUM_NEURONS
    )  # coefficients for producing doublets

    # Define the refractory period (assumed to be 2 ms for all neurons)
    # refractory_time_ms = 2  # in ms

    # Create a list to store parameters for each neuron
    neuron_list = []

    for i in range(num_neurons):
        D_soma_unit = soma_diameters[i]  # Soma diameter in meters

        # Calculate dependent parameters using empirical relationships
        S_unit = 5.5e-3 * D_soma_unit  # (Neuron surface area) in square meters [m^2]
        I_Rheobase_Ampere = 7.8e2 * np.pow(D_soma_unit, 2.52)  # Rheobase current [A]
        Doublet_current_Ampere = (
            I_Rheobase_Ampere * doublet_currents_coefficient[i]
        )  # Doublet current threshold [A]
        R_unit = 1.68e-10 * np.pow(S_unit, -2.43)  # Input resistance [Ω]
        C_unit = 7.9e-5 * D_soma_unit  # Membrane capacitance [F]
        tau_unit = 2.3e-9 * np.pow(
            D_soma_unit, -1.48
        )  # 7.9e-5 * D_soma_unit * R_unit  # Membrane time constant [s]
        t_ref_unit = (
            0.3  # decreased refractory time to make it work
            * 0.2
            * 2.7e-8
            * np.pow(D_soma_unit, -1.51)
        )  # (Refractory time) in [s]

        # adjust units
        I_th = I_Rheobase_Ampere * 1e9  # Rheobase current [nA]
        Doublet_Current = Doublet_current_Ampere * 1e9  # Double current threshold [nA]
        R = R_unit * 1e-6  # Input resistance [MΩ]
        C = C_unit * 1e9  # Membrane capacitance [nF]
        tau = tau_unit * 1e3  # Membrane time constant [ms]
        t_ref = t_ref_unit * 1e3  # (Refractory time) in [ms]
        D_soma = D_soma_unit * 1e6  # Soma diameter in [μm]
        S_soma = S_unit * 1e12

        # Calculate leak conductance (g_L = C / tau or 1 / R)
        g_L = 1 / R * 10  # in μS (since C is in nF and tau is in ms)

        # Set other parameters
        V_rest = -70  # Resting potential in mV
        V_th = -50  # Threshold potential in mV
        V_reset = -70  # Reset potential in mV
        V_init = V_rest  # Initial potential in mV
        E_L = -70.0  # leak reversal potential in mV
        gain_leak = leak_array[i]  # Gain parameter leakage
        gain_exc = leak_array[i]  # Gain parameter excitability

        # Store parameters for the neuron
        neuron_list.append(
            {
                "number": i,
                "D_soma": D_soma,
                "S_soma": S_soma,
                "R": R,  # Input resistance [MΩ]
                "C": C,  # Membrane capacitance [nF]
                "tau_m": tau,  # Membrane time constant [ms]
                "I_th": I_th,  # Rheobase current [nA]
                "Doublet_current": Doublet_Current,  # Double current thresholds [nA]
                "g_L": g_L,  # Leak conductance [μS]
                "V_rest": V_rest,  # Resting potential [mV]
                "V_th": V_th,  # Threshold potential [mV]
                "V_reset": V_reset,  # Reset potential [mV]
                "V_init": V_init,  # Initial potential [mV]
                "E_L": E_L,  # leak reversal potential [mV]
                "gain_leak": gain_leak,
                "gain_exc": gain_exc,
                "tref": t_ref,  # Refractory period [ms]
                "T": T,  # Total duration of simulation [ms]
                "dt": dt,  # Simulation time step [ms]
                "range_t": np.arange(
                    0, T, dt
                ),  # Vector of discretized time points [ms]
            }
        )

    # Convert the sorted list to a dictionary for compatibility
    # pars = {i: neuron_list[i] for i in range(num_neurons)}

    return neuron_list
