import numpy as np
from neuron import Neuron
from typing import Tuple
from simulation.simulate import TimestepSimulation

##MODEL WITH VARIABLE RESET VOLTAGE AND Rheobase limit + Decaying Doublet blocking##


class LIF_Model3v2(TimestepSimulation):

    @staticmethod
    def simulate_neuron(
        sim_time: np.float64, timestep: np.float64, neuron: Neuron, Iinj: np.array
    ) -> Tuple[np.array, np.array, np.array, np.array]:
        """
        Simulate the LIF dynamics with external input current

        Args:
        neuron       : Neuron object containing parameters
        Iinj       : input current [nA]. The injected current here can be a value
                    or an array

        Returns:
        rec_v      : membrane potential
        rec_sp     : spike times
        inhib_trace: inhibition decay factor over time
        reset_trace: reset voltage trace over time
        """

        simulation_steps = len(np.arange(0, sim_time, timestep))

        # Initialize voltage
        v = np.zeros(simulation_steps)
        v[0] = neuron.V_init_mV
        V_reset_it = neuron.V_reset_mV

        inhib_trace = np.zeros(simulation_steps)
        reset_trace = np.full(simulation_steps, np.nan)

        # Set current time course
        # Iinj = Iinj * np.ones(sim_steps)

        # Loop over time
        rec_spikes = []  # record spike times
        tr = 0.0  # the count for refractory duration
        last_spike_counter = (
            100 / timestep
        )  # time since spike, used for doublet interval (3-10ms).

        peak_voltage = 20
        excitability = 0  # Track if neuron has increased excitability, 1 = increased, 0 = normal, -1 = post doublet
        doublet_block = 0.0

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

            elif v[it] >= neuron.V_th_mV:
                ## ---- DOUBLET ---- ##
                if (
                    last_spike_counter < 10 / timestep
                    and Iinj[it] >= neuron.I_rheobase
                    and doublet_block
                    < 0.5  # Adjust this threshold on how long to block for, could be neuron-dependant..
                ):
                    rec_spikes.append(it)
                    peak_voltage = 18  # 18mV for doublet
                    v[it] = peak_voltage
                    # set new refractory time : double normal time.
                    tr = neuron.tref * 2 / timestep
                    decay_steps = tr
                    last_spike_counter = 0.0

                    doublet_block = 1.0
                    V_reset_it = neuron.V_reset_mV - 10
                    # Doublet block does not impact reset voltage.
                    excitability = -1

                ## ---- NORMAL SPIKE ---- ##
                else:
                    rec_spikes.append(it)

                    peak_voltage = 20
                    v[it] = peak_voltage  # 20mV more biologically accurate
                    tr = neuron.tref / timestep  # set refractory time
                    decay_steps = tr

                    V_reset_it = neuron.calculate_v_reset(Iinj[it])
                    doublet_block = 1.0

                    if V_reset_it > neuron.V_reset_mV:
                        excitability = 1
                    else:
                        excitability = 0

                    last_spike_counter = 0.0

            if (
                last_spike_counter > 2 / timestep and excitability == 1
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

            doublet_block *= np.exp(-timestep / 500.0)

            inhib_trace[it] = doublet_block
            reset_trace[it] = V_reset_it

        # Get spike times in ms
        rec_spikes = np.array(rec_spikes) * timestep
        # print(doub_count)

        return v, rec_spikes, inhib_trace, reset_trace
