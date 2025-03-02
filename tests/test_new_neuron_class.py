import unittest
import numpy as np
from models_legacy.mn_creation import caillet_quadratic
from neuron import NeuronFactory, soma_diameter_vector

# Import original and refactored functions
# from original_module import (
#     run_lif as original_run_lif,
# )
# from refactored_module import (
#     run_lif as refactored_run_lif,
# )

COMPARE_FIELDS = {
    "number": "number",
    "D_soma": "D_soma",
    "S_soma": "S_soma",
    "R": "R_Mohm",
    "C": "C_nF",
    "tau_m": "tau_ms",
    "I_th": "I_rheobase",
    "Doublet_current": "Rheobase_threshold",
    "g_L": "g_L",
    "V_rest": "V_rest_mV",
    "V_th": "V_th_mV",
    "V_reset": "V_reset_mV",
    "V_init": "V_init_mV",
    "E_L": "E_L_mV",
    "gain_leak": "gain_leak",
    "gain_exc": "gain_exc",
    "tref": "tref",
    # 'T',
    # 'dt',
    # 'range_t'
}


def compare_neurons(old_n, new_n):
    for old_field, new_field in COMPARE_FIELDS.items():
        if not np.isclose(old_n[old_field], getattr(new_n, new_field)):
            a = old_n[old_field]
            b = getattr(new_n, new_field)
            print(f"Value mismatch: {old_field}:{new_field}")
            print(a)
            print(b)
            return False
    return True


class TestNeuronSimulation(unittest.TestCase):
    """Test class"""

    def setUp(self):
        """Set up initial neuron lists for both versions."""
        # Neurons to compare in test
        compare_neurons = [5, 50, 150, 200, 275]

        # Generate list of old neurons
        T = 1000  # Simulation Time [ms]
        DT = 0.1  # Time step in [ms]
        NUM_NEURONS = 300  # Number of Neurons simulated
        self.original_neurons = caillet_quadratic(T, DT, NUM_NEURONS)
        self.original_neurons = [self.original_neurons[i] for i in compare_neurons]

        # Generate list of new neurons
        soma_vector = soma_diameter_vector(total_neurons=NUM_NEURONS)[compare_neurons]
        self.refactored_neurons = [
            NeuronFactory.create_neuron(
                number=n, soma_diameter=d, num_total_neurons=NUM_NEURONS
            )
            for n, d in zip(compare_neurons, soma_vector)
        ]

    def test_get_neurons_equivalence(self):
        """Ensure that the neuron lists from both versions are identical."""
        for index, (old_neuron, new_neuron) in enumerate(
            zip(self.original_neurons, self.refactored_neurons)
        ):
            print(f"Comparing neuron: {index}")
            self.assertTrue(
                compare_neurons(
                    old_neuron,
                    new_neuron,
                ),
                "The neuron lists do not match!",
            )

    # def test_run_lif_equivalence(self):
    #     """Ensure that the simulation output from both versions is identical."""
    #     original_output = original_run_lif(self.original_neurons)
    #     refactored_output = refactored_run_lif(self.refactored_neurons)

    #     if isinstance(original_output, np.ndarray) and isinstance(
    #         refactored_output, np.ndarray
    #     ):
    #         np.testing.assert_array_almost_equal(
    #             original_output, refactored_output, decimal=6
    #         )
    #     else:
    #         self.assertEqual(
    #             original_output, refactored_output, "Simulation outputs do not match!"
    #         )


if __name__ == "__main__":
    unittest.main()
