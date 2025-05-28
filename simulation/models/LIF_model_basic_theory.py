import numpy as np
from neuron import Neuron
from typing import Tuple
from simulation.simulate import TimestepSimulation

## Used to create theoretical output plot for thesis, not accurate modelling!##


class LIF_Model_basic_theory(TimestepSimulation):

    @staticmethod
    def simulate_neuron(
        sim_time: np.float64, timestep: np.float64, neuron: Neuron, Iinj: np.array
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

        simulation_steps = len(Iinj)

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
                # Gradually decay voltage toward reset using exponential decay
                tau_refractory = 8  # you can tune this time constant (in time steps)
                v[it] = v[it - 1] + (neuron.V_reset_mV - v[it - 1]) / tau_refractory
                tr -= 1  # decrement refractory counter
                if len(rec_spikes) < 2:
                    if tr < 0:  # increased excitation for doublet.
                        v[it] = neuron.V_reset_mV / 2

            elif v[it] >= neuron.V_th_mV:  # if voltage over threshold
                rec_spikes.append(it)  # record spike event
                if len(rec_spikes) == 2:

                    v[it] = -10  # spike voltage for doublet
                    tr = neuron.tref / timestep * 5
                else:
                    v[it] = 0  # spike voltage
                    # v[it] = neuron.V_reset_mV  # reset voltage
                    tr = neuron.tref / timestep  # set refractory time

            # Calculate the increment of the membrane potential
            dv = (
                -(neuron.gain_leak) * (v[it] - neuron.E_L_mV)
                + (neuron.gain_exc) * (Iinj[it] * neuron.R_Mohm)
            ) * (timestep / neuron.tau_ms)

            # Update the membrane potential [mv]
            v[it + 1] = v[it] + dv

        # Get spike times in ms
        rec_spikes = np.array(rec_spikes) * timestep
        # print(doub_count)

        return v, rec_spikes
