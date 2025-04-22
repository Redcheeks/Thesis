import numpy as np
from neuron import Neuron
from typing import Tuple
from simulation.simulate import TimestepSimulation

##MODEL WITH VARIABLE RESET VOLTAGE AND RENSHAW##


class LIF_Model3(TimestepSimulation):

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

        inhib_decay_factor = 0.0

        for it in range(simulation_steps - 1):

            if tr > 0:  # check if in refractory period
                # TODO: could be a nice curve down rather than a steep drop
                v[it] = V_reset_it  # set voltage to reset
                reset_trace[it] = V_reset_it

                tr = tr - 1  # reduce running counter of refractory period

            elif v[it] >= neuron.V_th_mV:
                ## ---- DOUBLET ---- ##
                if last_spike_counter < 10 / timestep:
                    v[it] = 18  # 18mV for doublet
                    rec_spikes.append(it)
                    # Removed line: v[it - 1] = 0
                    last_spike_counter = 0.0
                    inhib_decay_factor += 1.0
                    V_reset_it = neuron.calculate_v_reset_MODEL3(
                        Iinj[it], inhib_decay_factor
                    )
                    reset_trace[it] = V_reset_it
                    # v[it] = V_reset_it
                    tr = neuron.tref * 2 / timestep
                ## ---- NORMAL SPIKE ---- ##
                else:
                    v[it] = 20  # 20mV biologically accurate?
                    rec_spikes.append(it)
                    V_reset_it = neuron.calculate_v_reset_MODEL3(
                        Iinj[it], inhib_decay_factor
                    )
                    reset_trace[it] = V_reset_it
                    # v[it] = V_reset_it
                    tr = neuron.tref / timestep
                    last_spike_counter = 0.0

            # Calculate the increment of the membrane potential
            dv = (
                -(neuron.gain_leak) * (v[it] - neuron.E_L_mV)
                + (neuron.gain_exc) * (Iinj[it] * neuron.R_Mohm)
            ) * (timestep / neuron.tau_ms)

            # Update the membrane potential [mv]
            v[it + 1] = v[it] + dv
            last_spike_counter += 1

            inhib_decay_factor *= np.exp(-timestep / 500.0)

            inhib_trace[it] = inhib_decay_factor
            # Removed the line: reset_trace[it] = V_reset_it

        for i in range(1, simulation_steps):
            if np.isnan(reset_trace[i]):
                reset_trace[i] = reset_trace[i - 1]

        # Get spike times in ms
        rec_spikes = np.array(rec_spikes) * timestep
        # print(doub_count)

        return v, rec_spikes, inhib_trace, reset_trace
