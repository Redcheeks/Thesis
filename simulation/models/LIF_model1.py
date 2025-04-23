import numpy as np
from neuron import Neuron
from typing import Tuple
from simulation.simulate import TimestepSimulation

##MODEL BASED ON RHEOBASE CURRENT LIMIT##


class LIF_Model1(TimestepSimulation):

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
        renshaw_trace : trace of renshaw inhibition state over time
        """

        simulation_steps = len(np.arange(0, sim_time, timestep))

        # Initialize voltage
        v = np.zeros(simulation_steps)
        v[0] = neuron.V_init_mV

        renshaw_trace = np.zeros(simulation_steps)

        # Set current time course
        # Iinj = Iinj * np.ones(sim_steps)

        # Loop over time
        rec_spikes = []  # record spike times
        tr = 0.0  # the count for refractory duration
        last_spike_counter = (
            100 / timestep
        )  # time since spike, used for doublet interval (3-10ms).
        renshaw_inhib = False  # is renshaw cell inhibiting
        renshaw_reset = 200 / timestep
        relax_counter = (
            renshaw_reset  # used to check for relaxation period for renshaw state.
        )
        peak_voltage = 20
        doub_count = 0  # For testing purposes

        for it in range(simulation_steps - 1):

            renshaw_trace[it] = float(renshaw_inhib)

            if relax_counter > renshaw_reset:
                renshaw_inhib = False

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
                    v[it + 1] = neuron.V_reset_mV  # lock in the final decay value
                    continue

            ## ---- NORMAL SPIKE ---- ##
            elif v[it] >= neuron.V_th_mV:  # if voltage over threshold
                rec_spikes.append(it)  # record spike event
                last_spike_counter = 0.0

                if (
                    relax_counter > renshaw_reset
                ):  # check if neuron was relaxed prior to this spike
                    renshaw_inhib = False
                else:
                    renshaw_inhib = True

                peak_voltage = 20
                v[it] = peak_voltage  # 20mV more biologically accurate
                tr = neuron.tref / timestep  # set refractory time
                decay_steps = tr

                relax_counter = 0.0
                # v[it] = neuron.V_reset_mV  # reset voltage

            ## ---- DOUBLET ---- ##
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
                peak_voltage = 18
                v[it] = peak_voltage
                # v[it] = neuron.V_reset_mV  # reset voltage
                tr = (
                    neuron.tref * 2 / timestep
                )  # set new refractory time : double normal time.
                decay_steps = tr

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

        return v, rec_spikes, renshaw_trace
