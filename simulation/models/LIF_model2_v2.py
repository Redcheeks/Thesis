import numpy as np
from neuron import Neuron
from typing import Tuple
from simulation.simulate import TimestepSimulation

##MODEL WITH ADP/ EXCITABILITY BOOST & DECAY FOR FULL DELAYED DEPOLARIZATION SIM##
# Uncomplete model exploring an alternative approach without the adjusted reset voltage.


class LIF_Model2v2(TimestepSimulation):

    @staticmethod
    def simulate_neuron(
        sim_time: np.float64, timestep: np.float64, neuron: Neuron, Iinj: np.array
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Deprecated Model2: Simulate the Basic LIF dynamics (WITH EXCITABILITY BOOST FACTOR) with external input current

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

        simulation_steps = len(np.arange(0, sim_time, timestep))

        # Initialize voltage
        v = np.zeros(simulation_steps)
        v[0] = neuron.V_init_mV
        V_reset_it = neuron.V_reset_mV

        # Loop over time
        rec_spikes = []  # record spike times
        tr = 0.0  # the count for refractory duration
        last_spike_counter = (
            100 / timestep
        )  # time since spike, used for doublet interval (3-10ms).

        peak_voltage = 20

        # --- Parameters for ADP boost and decay
        adp_boost = 0.0  # Starts at zero
        adp_tau = 5.0  # Time constant in ms
        adp_peak = 2.0  # Maximum possible boost in mV
        epsilon = 0.1  # Safety margin to not hit threshold alone
        scaling_factor = 2.0  # mV per nA above rheobase
        adp_timer = 0.0  # intilize boost window timer

        boost_trace = np.zeros(simulation_steps)

        for it in range(simulation_steps - 1):

            if (
                tr > 0
            ):  # check if in refractory period, smooth voltage decay towards reset
                progress = (decay_steps - tr) / decay_steps
                sharpness = 7  # controls steepness of early drop
                curve_factor = 1 - np.exp(-sharpness * progress)

                v[it] = peak_voltage - (peak_voltage - neuron.V_reset_mV) * curve_factor
                tr -= 1  # decrement refractory counter

                if tr < 1:
                    # Start ADP phase
                    adp_duration = int(10 / timestep)  # 10ms ADP window
                    adp_timer = adp_duration
                    rheo_diff = max(0, Iinj[it] - neuron.I_rheobase)
                    scaled_boost = min(adp_peak, rheo_diff * scaling_factor)
                    adp_boost = scaled_boost
                    boost_trace[it] = adp_boost

                v[it + 1] = v[it]  # carry forward voltage even if refractory ends here
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
                    # Reset ADP effect after doublet
                    adp_boost = 0.0
                    adp_timer = 0
                    boost_trace[it] = adp_boost

                    # After doublet reset voltage is even lower, 10 is an arbitrary value to simulate the intensified AHP!!

                ## ---- NORMAL SPIKE ---- ##
                else:
                    rec_spikes.append(it)  # record spike event

                    peak_voltage = 20
                    v[it] = peak_voltage  # 20mV more biologically accurate
                    tr = neuron.tref / timestep  # set refractory time
                    decay_steps = tr

                    last_spike_counter = 0.0
                    # At spike

                    adp_duration = int(10 / timestep)  # 10ms worth of boost
                    adp_timer = adp_duration

                    # Scale boost based on input proximity to rheobase
                    rheo_diff = max(0, Iinj[it] - neuron.I_rheobase)
                    scaled_boost = min(
                        adp_peak, rheo_diff * scaling_factor
                    )  # e.g., 2.0 per nA
                    adp_boost = scaled_boost
                    boost_trace[it] = adp_boost

            # Calculate the increment of the membrane potential
            dv = (
                -(neuron.gain_leak) * (v[it] - neuron.E_L_mV)
                + (neuron.gain_exc) * (Iinj[it] * neuron.R_Mohm)
            ) * (timestep / neuron.tau_ms)

            # Update the membrane potential, normal or boosted [mv]
            if adp_timer > 0:
                v[it + 1] = v[it] + dv + adp_boost
                adp_boost *= np.exp(-timestep / adp_tau)
                boost_trace[it] = adp_boost
                adp_timer -= 1
            else:
                v[it + 1] = v[it] + dv

            last_spike_counter += 1

        # Get spike times in ms
        rec_spikes = np.array(rec_spikes) * timestep
        # print(doub_count)

        return v, rec_spikes, boost_trace
