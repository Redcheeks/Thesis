import numpy as np
from neuron import Neuron
from typing import Tuple
from simulation.simulate import TimestepSimulation

##Complete Model 2 WITH VARIABLE RESET VOLTAGE + RESET IF NO DOUBLET (Delayed Depolarization bump as a "square wave")##
## Also has post doublet depression thorugh increased t_ref and decreased V_reset.


class LIF_Model2v3(TimestepSimulation):

    @staticmethod
    def simulate_neuron(
        sim_time: np.float64, timestep: np.float64, neuron: Neuron, Iinj: np.array
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Model2: Simulate the Basic LIF dynamics WITH ADP with external input current

        Args:
        sim_time    : Simulation run-time (ms)
        timestep    : time step (ms)
        neuron      : Neuron object containing parameters
        Iinj        : input current [nA]. The injected current should be an array of the same length as sim_time/dt

        Returns:
        rec_v           : recorded membrane potential [array]
        rec_sp          : recorded spike times [array]
        reset_trace     : reset voltage over time [array]
        """

        simulation_steps = len(Iinj)

        # Initialize voltage
        v = np.zeros(simulation_steps)
        v[0] = neuron.V_init_mV
        V_reset_it = neuron.V_reset_mV

        reset_trace = np.full(simulation_steps, np.nan)

        # Loop over time
        rec_spikes = []  # record spike times
        tr = 0.0  # the count for refractory duration
        last_spike_counter = (
            100 / timestep
        )  # time since spike, used for doublet interval (3-10ms).

        peak_voltage = 20
        excitability = 0  # Track if neuron has increased excitability, 1 = increased, 0 = normal, -1 = post doublet

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
                if last_spike_counter < 10 / timestep:
                    rec_spikes.append(it)  # record spike event
                    peak_voltage = 18  # 18mV for doublet
                    v[it] = peak_voltage

                    # set new refractory time : double normal time.
                    tr = neuron.tref * 2 / timestep
                    decay_steps = tr
                    last_spike_counter = 0.0

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

                    # ------- Calculate new reset voltage based on how close to rheobase current
                    # simulates delayed depolarization bump

                    V_reset_it = neuron.calculate_v_reset(Iinj[it])
                    if V_reset_it > neuron.V_reset_mV:
                        excitability = 1
                    else:
                        excitability = 0

                    last_spike_counter = 0.0

            ## SOURCES:
            # Kernell 1964 - ADP hump lasts about 2.5-6ms, choose 4ms. Future work: let this be neuron dependant.
            # Ionic model Purvis and Butera shows the ADP bump lasting closer to 2ms
            # From Kudina 1989 - increased exciytability for 20msec post spike for neurons that are able to produce doublets???
            # From Halonen 1977 - the ISI post doublet is approx 1.5 times the ISI of previous spikes. Assumed to be due to prolonged Afterhyperpolarization period. (Kudina 2013, Kudina 2010 and many more give similar numbers!)
            if (
                excitability == 1 and last_spike_counter > 8 / timestep
            ):  # Check if doublet didnt occur from delayed. depol. bump => then return to normal excitability levels
                excitability = 0
                v[it] = v[it] + (
                    neuron.V_reset_mV - V_reset_it
                )  # Instant decay of delayed depolarization bump
                V_reset_it = neuron.V_reset_mV
            # Post-doublet depression ends naturally, this might not be needed..
            # elif (
            #     excitability == -1 and last_spike_counter > 50 / timestep
            # ):  # decreased excitability recovers after 50-200msec.. (Brooks 1950)
            #     excitability = 0
            #     v[it] = v[it] + (
            #         neuron.V_reset_mV - V_reset_it
            #     )  # Instant decay of decreased excitability post doublet.
            #     V_reset_it = neuron.V_reset_mV

            # Calculate the increment of the membrane potential
            dv = (
                -(neuron.gain_leak) * (v[it] - neuron.E_L_mV)
                + (neuron.gain_exc) * (Iinj[it] * neuron.R_Mohm)
            ) * (timestep / neuron.tau_ms)

            # Update the membrane potential [mv]
            v[it + 1] = v[it] + dv
            last_spike_counter += 1
            reset_trace[it] = V_reset_it
        # Get spike times in ms
        rec_spikes = np.array(rec_spikes) * timestep
        # print(doub_count)

        return v, rec_spikes, reset_trace
