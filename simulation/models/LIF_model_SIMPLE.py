import numpy as np
from neuron import Neuron
from typing import Tuple
from simulation.simulate import TimestepSimulation

##Most Simple LIF with no doublets##


class LIF_SIMPLE(TimestepSimulation):

    @staticmethod
    def simulate_neuron(
        sim_time: np.float64, dt: np.float64, neuron: Neuron, Iinj: np.array
    ) -> Tuple[np.array, np.array]:
        """
        Simulate the Basic LIF dynamics with external input current

        Args:
        neuron       : Neuron object containing parameters
        Iinj       : input current [nA]. The injected current here can be a value
                    or an array

        Returns:
        rec_v      : membrane potential
        rec_sp     : spike times
        """

        simulation_steps = len(np.arange(0, sim_time, dt))

        # Initialize voltage
        v = np.zeros(simulation_steps)
        v[0] = neuron.V_init_mV

        # Set current time course
        # Iinj = Iinj * np.ones(sim_steps)

        # Loop over time
        rec_spikes = []  # record spike times
        tr = 0.0  # the count for refractory duration

        for it in range(simulation_steps - 1):

            if tr > 0:  # check if in refractory period
                v[it + 1] = neuron.V_reset_mV
                tr = tr - 1  # reduce running counter of refractory period

            elif v[it] >= neuron.V_th_mV:  # if voltage over threshold
                rec_spikes.append(it)  # record spike event
                v[it + 1] = neuron.V_reset_mV  # reset voltage
                tr = neuron.tref / dt  # set refractory time

            # Calculate the increment of the membrane potential
            dv = (
                -(neuron.gain_leak) * (v[it] - neuron.E_L_mV)
                + (neuron.gain_exc) * (Iinj[it] * neuron.R_Mohm)
            ) * (dt / neuron.tau_ms)

            # Update the membrane potential [mv]
            v[it + 1] = v[it] + dv

        # Get spike times in ms
        rec_spikes = np.array(rec_spikes) * dt
        # print(doub_count)

        return v, rec_spikes
