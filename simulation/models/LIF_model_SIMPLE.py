import numpy as np
from neuron import Neuron
from typing import Tuple
from simulation.simulate import TimestepSimulation

##Most Simple LIF model with no doublets, based on Neuromatch (W2D3)##


class LIF_SIMPLE(TimestepSimulation):

    @staticmethod
    def simulate_neuron(
        sim_time: np.float64, dt: np.float64, neuron: Neuron, Iinj: np.array
    ) -> Tuple[np.array, np.array]:
        """
        Simulate the Basic LIF dynamics with external input current

        Args:
        sim_time    : Simulation run-time (ms)
        dt          : time step (ms)
        neuron      : Neuron object containing parameters
        Iinj        : input current [nA]. The injected current should be an array of the same length as sim_time/dt

        Returns:
        rec_v      : membrane potential array
        rec_sp     : spike times array
        """

        simulation_steps = len(np.arange(0, sim_time, dt))

        # Initialize voltage
        v = np.zeros(simulation_steps)
        v[0] = neuron.V_init_mV

        # Loop over time
        rec_spikes = []  # record spike times
        tr = 0.0  # the count for refractory duration

        for it in range(simulation_steps - 1):

            if tr > 0:  # check if in refractory period
                v[it] = neuron.V_reset_mV  # keep voltage at reset
                tr = tr - 1  # reduce running counter of refractory period

            elif v[it] >= neuron.V_th_mV:  # if voltage over threshold
                rec_spikes.append(it)  # record spike event
                v[it] = neuron.V_reset_mV  # reset voltage
                tr = neuron.tref / dt  # set refractory time

            # Calculate the increment of the membrane potential for each time-step
            dv = (
                -(neuron.gain_leak) * (v[it] - neuron.E_L_mV)
                + (neuron.gain_exc) * (Iinj[it] * neuron.R_Mohm)
            ) * (dt / neuron.tau_ms)

            # Update the membrane potential [mv]
            v[it + 1] = v[it] + dv

        # Get spike times in ms
        rec_spikes = np.array(rec_spikes) * dt

        return v, rec_spikes
