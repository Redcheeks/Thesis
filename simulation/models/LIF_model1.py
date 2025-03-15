import numpy as np
from neuron import Neuron
from typing import Tuple
from simulation.simulate import TimestepSimulation

##MODEL BASED ON RHEOBASE CURRENT LIMIT##


class LIF_Model1(TimestepSimulation):

    @staticmethod
    def simulate_neuron(
        sim_time: np.float64, timestep: np.float64, neuron: Neuron, Iinj: np.array
    ) -> Tuple[np.array, np.array]:
        """
        Simulate the LIF dynamics with external input current

        Args:
        neuron       : Neuron object containing parameters
        Iinj       : input current [nA]. The injected current here can be a value
                    or an array

        Returns:
        rec_v      : membrane potential
        rec_sp     : spike times
        """

        simulation_steps = len(np.arange(0, sim_time, timestep))

        # Initialize voltage
        v = np.zeros(simulation_steps)
        v[0] = neuron.V_init_mV

        # Set current time course
        # Iinj = Iinj * np.ones(sim_steps)

        # Loop over time
        rec_spikes = []  # record spike times
        tr = 0.0  # the count for refractory duration
        last_spike_counter = (
            100 / timestep
        )  # time since spike, used for doublet interval (3-10ms).
        renshaw_inhib = False  # is renshaw cell inhibiting
        renshaw_reset = 400 / timestep
        relax_counter = (
            renshaw_reset  # used to check for relaxation period for renshaw state.
        )
        doub_count = 0  # For testing purposes

        for it in range(simulation_steps - 1):

            if relax_counter > renshaw_reset:
                renshaw_inhib = False

            if tr > 0:  # check if in refractory period
                v[it] = neuron.V_reset_mV  # set voltage to reset
                tr = tr - 1  # reduce running counter of refractory period

            elif v[it] >= neuron.V_th_mV:  # if voltage over threshold
                rec_spikes.append(it)  # record spike event
                last_spike_counter = 0.0

                if (
                    relax_counter > renshaw_reset
                ):  # check if neuron was relaxed prior to this spike
                    renshaw_inhib = False
                else:
                    renshaw_inhib = True

                relax_counter = 0.0
                v[it] = neuron.V_reset_mV  # reset voltage
                tr = neuron.tref / timestep  # set refractory time

            elif (
                (3 / timestep)
                < last_spike_counter
                < (10 / timestep)  # doublet interval
                and Iinj[it] >= neuron.I_rheobase  # current threshold
                and renshaw_inhib == False  # renshaw cell is not in inhibition state
            ):
                rec_spikes.append(it)  # record spike event
                relax_counter = 0.0
                renshaw_inhib = True
                v[it - 1] = (
                    neuron.V_th_mV + 20
                )  ##ONLY FOR MAKING DOUBLETS MORE VISIBLE!!
                v[it] = neuron.V_reset_mV  # reset voltage
                tr = (
                    neuron.tref * 2 / timestep
                )  # set new refractory time : double normal time.
                doub_count += 1

            # Calculate the increment of the membrane potential
            dv = (
                -(neuron.gain_leak) * (v[it] - neuron.E_L_mV)
                + (neuron.gain_exc) * (Iinj[it] * neuron.R_Mohm)
            ) * (timestep / neuron.tau_ms)

            # Update the membrane potential [mv]
            v[it + 1] = v[it] + dv
            relax_counter += 1
            last_spike_counter += 1

        # Get spike times in ms
        rec_spikes = np.array(rec_spikes) * timestep
        # print(doub_count)

        return v, rec_spikes
