import numpy as np
from neuron import Neuron
from typing import Tuple
from simulation.simulate import TimestepSimulation

##MODEL WITH VARIABLE RESET VOLTAGE + growing Inhibition decay factor + rheobase threshold for doublet##
# inhibition affects the variable reset voltage calculated in neuron.py class.
# ALSO!! rheobase requirement for doublets


class LIF_Model3(TimestepSimulation):

    @staticmethod
    def simulate_neuron(
        sim_time: np.float64, timestep: np.float64, neuron: Neuron, Iinj: np.array
    ) -> Tuple[np.array, np.array, np.array, np.array]:
        """
        Model3: Simulate the Basic LIF dynamics + ADP & I_SK with external input current

        Args:
        sim_time    : Simulation run-time (ms)
        timestep    : time step (ms)
        neuron      : Neuron object containing parameters
        Iinj        : input current [nA]. The injected current should be an array of the same length as sim_time/dt

        Returns:
        rec_v           : recorded membrane potential [array]
        rec_sp          : recorded spike times [array]
        inhib_trace     : trace of inhibition over time [array]
        reset_trace     : reset voltage over time [array]

        """

        simulation_steps = len(Iinj)

        # Initialize voltage
        v = np.zeros(simulation_steps)
        v[0] = neuron.V_init_mV
        V_reset_it = neuron.V_reset_mV

        inhib_trace = np.zeros(simulation_steps)
        reset_trace = np.full(simulation_steps, np.nan)

        # Loop over time
        rec_spikes = []  # record spike times
        tr = 0.0  # the count for refractory duration
        last_spike_counter = (
            100 / timestep
        )  # time since spike, used for doublet interval (3-10ms).

        peak_voltage = 20
        excitability = 0  # Track if neuron has increased excitability, 1 = increased, 0 = normal, -1 = post doublet
        inhib_decay_factor = 0.0

        # Decay is based on Purvis & Butera I_SK current: K2=0.04 -> tau = 100, K2 = decay rate constant representing calcium uptake and diffusion (ms)
        tau = 100  # 100.0  # Adjust this value for your desired decay time

        for it in range(simulation_steps - 1):

            if (
                tr > 0
            ):  # check if in refractory period, smooth voltage decay towards reset
                progress = (decay_steps - tr) / decay_steps
                sharpness = 7  # controls steepness of early drop
                curve_factor = 1 - np.exp(-sharpness * progress)

                if excitability == -1:  # Go directly towards final reset
                    v[it] = peak_voltage - (peak_voltage - V_reset_it) * curve_factor
                else:  # go towards normal reset, then jump to excitment for delayed depolarization. Currently set to happen at the end of absolute refractory period...
                    v[it] = (
                        peak_voltage - (peak_voltage - neuron.V_reset_mV) * curve_factor
                    )

                tr -= 1  # decrement refractory counter
                if (
                    tr > 1
                ):  # after the last step we need to run the incremental potential for the next it.
                    continue
                else:
                    v[it + 1] = V_reset_it  # lock in the final decay value

                    continue

            elif v[it] >= neuron.V_th_mV:  # if voltage over threshold
                ## ---- DOUBLET ---- ##
                if last_spike_counter < 10 / timestep and Iinj[it] >= neuron.I_rheobase:
                    rec_spikes.append(it)  # record spike event
                    peak_voltage = 18  # 18mV for doublet
                    v[it] = peak_voltage
                    # set new refractory time : double normal time.
                    tr = neuron.tref * 2 / timestep
                    decay_steps = tr
                    last_spike_counter = 0.0

                    inhib_decay_factor += 0.5 * (1.0 - inhib_decay_factor)
                    # After doublet reset voltage is even lower, 10 is an arbitrary value to simulate the intensified AHP!!
                    V_reset_it = neuron.V_reset_mV - 10

                    excitability = -1

                ## ---- NORMAL SPIKE ---- ##
                else:
                    rec_spikes.append(it)  # record spike event

                    peak_voltage = 20
                    v[it] = peak_voltage  # 20mV more biologically accurate
                    tr = neuron.tref / timestep  # set refractory time
                    decay_steps = tr

                    # ------- Calculate new reset voltage based on how close to rheobase current, taking inhibition into account!
                    # simulates delayed depolarization bump and its intensity decay over time.

                    V_reset_it = neuron.calculate_v_reset_MODEL3(
                        Iinj[it], inhib_decay_factor
                    )
                    inhib_decay_factor += 0.5 * (1.0 - inhib_decay_factor)

                    if V_reset_it > neuron.V_reset_mV:
                        excitability = 1
                    else:
                        excitability = 0

                    last_spike_counter = 0.0

            ## See comments in LIF_Model2v3 for more info on below code.
            if (
                excitability == 1 and last_spike_counter > 8 / timestep
            ):  # Check if doublet didnt occur from delayed. depol. bump => then return to normal excitability levels
                excitability = 0
                v[it] = v[it] + (
                    neuron.V_reset_mV - V_reset_it
                )  # Instant decay of delayed depolarization bump
                V_reset_it = neuron.V_reset_mV
            # Calculate the increment of the membrane potential
            dv = (
                -(neuron.gain_leak) * (v[it] - neuron.E_L_mV)
                + (neuron.gain_exc) * (Iinj[it] * neuron.R_Mohm)
            ) * (timestep / neuron.tau_ms)

            # Update the membrane potential [mv]
            v[it + 1] = v[it] + dv
            last_spike_counter += 1

            inhib_decay_factor *= np.exp(-timestep / tau)

            inhib_trace[it] = inhib_decay_factor
            reset_trace[it] = V_reset_it

        # Get spike times in ms
        rec_spikes = np.array(rec_spikes) * timestep
        # print(doub_count)

        return v, rec_spikes, inhib_trace, reset_trace
