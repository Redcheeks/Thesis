import numpy as np
from neuron import Neuron
from typing import Tuple
from simulation.simulate import TimestepSimulation

##MODEL WITH VARIABLE RESET VOLTAGE AND EXCITABILITY DECAY FOR FULL DELKAYED DEPOLARIZATION##


class LIF_Model2(TimestepSimulation):

    @staticmethod
    def simulate_neuron(
        sim_time: np.float64, timestep: np.float64, neuron: Neuron, Iinj: np.array
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Simulate the LIF dynamics with external input current

        Args:
        neuron       : Neuron object containing parameters
        Iinj       : input current [nA]. The injected current here can be a value
                    or an array

        Returns:
        rec_v      : membrane potential
        rec_sp     : spike times
        reset_trace: reset voltage over time
        """

        simulation_steps = len(np.arange(0, sim_time, timestep))

        # Initialize voltage
        v = np.zeros(simulation_steps)
        v[0] = neuron.V_init_mV
        V_reset_it = neuron.V_reset_mV

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

        for it in range(simulation_steps - 1):

            if tr > 0:  # check if in refractory period
                progress = (decay_steps - tr) / decay_steps
                sharpness = 7  # controls steepness of early drop
                curve_factor = 1 - np.exp(-sharpness * progress)

                v[it] = peak_voltage - (peak_voltage - neuron.V_reset_mV) * curve_factor
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
                    peak_voltage = 18
                    v[it] = peak_voltage
                    # v[it] = neuron.V_reset_mV  # reset voltage
                    # set new refractory time : double normal time.
                    tr = neuron.tref * 2 / timestep
                    decay_steps = tr
                    last_spike_counter = 0.0

                    # After doublet reset voltage is even lower, 10 is an arbitrary value to simulate the intensified AHP!!
                    V_reset_it = neuron.V_reset_mV - 10

                ## ---- NORMAL SPIKE ---- ##
                else:
                    rec_spikes.append(it)  # record spike event

                    peak_voltage = 20
                    v[it] = peak_voltage  # 20mV more biologically accurate
                    tr = neuron.tref / timestep  # set refractory time
                    decay_steps = tr
                    # ------- Calculate new reset voltage based on how close to rheobase current
                    # !! makes possible for delayed depolarization bump, however this increase of excitability doesnt decay...
                    # very crude approximation that causes increased firing rate when near I_rheo but not close enough for doublets------ #
                    V_reset_it = neuron.calculate_v_reset(Iinj[it])

                    last_spike_counter = 0.0

            reset_trace[it] = V_reset_it

            # Calculate the increment of the membrane potential
            dv = (
                -(neuron.gain_leak) * (v[it] - neuron.E_L_mV)
                + (neuron.gain_exc) * (Iinj[it] * neuron.R_Mohm)
            ) * (timestep / neuron.tau_ms)

            # Update the membrane potential [mv]
            v[it + 1] = v[it] + dv
            last_spike_counter += 1

        # Get spike times in ms
        rec_spikes = np.array(rec_spikes) * timestep
        # print(doub_count)

        return v, rec_spikes, reset_trace
