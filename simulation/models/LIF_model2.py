import numpy as np
from neuron import Neuron
from typing import Tuple
from simulation.simulate import TimestepSimulation

##MODEL WITH VARIABLE RESET VOLTAGE##


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

        for it in range(simulation_steps - 1):

            if tr > 0:  # check if in refractory period
                # TODO: could be a nice curve down rather than a steep drop
                v[it] = V_reset_it  # set voltage to reset

                tr = tr - 1  # reduce running counter of refractory period

            elif v[it] >= neuron.V_th_mV:  # if voltage over threshold
                ## ---- DOUBLET ---- ##
                if last_spike_counter < 10 / timestep:

                    rec_spikes.append(it)  # record spike event
                    v[it] = 18  # 18mV for doublet
                    last_spike_counter = 0.0

                    V_reset_it = (
                        neuron.V_reset_mV - 10
                    )  #  # After doublet reset voltage is even lower, 10 is an arbitrary value..
                    # v[it + 1] = V_reset_it  # set voltage to reset
                    tr = (
                        neuron.tref * 2 / timestep
                    )  # set longer refractory time. Might not need this.
                ## ---- NORMAL SPIKE ---- ##
                else:
                    rec_spikes.append(it)  # record spike event

                    v[it] = 20  # 20mV biologically accurate?

                    # ------- Calculate new reset voltage based on how close to rheobase current ------ #
                    V_reset_it = neuron.calculate_v_reset(Iinj[it])
                    # v[it + 1] = V_reset_it  # set voltage to reset
                    tr = neuron.tref / timestep  # set refractory time
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
