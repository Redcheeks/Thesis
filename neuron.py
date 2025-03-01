from dataclasses import dataclass
import numpy as np
from typing import List

V_REST: np.float64 = -70.0  # Resting potential in mV
V_THRESHOLD: np.float64 = -50  # Threshold potential in mV
V_RESET: np.float64 = -70  # Reset potential in mV
V_INIT: np.float64 = -70  # Initial potential in mV
E_LEAK: np.float64 = -70.0  # leak reversal potential in mV

SOMA_DIAM_MIN_METER: np.float64 = 50e-6
SOMA_DIAM_MAX_METER: np.float64 = 100e-6


@dataclass(frozen=True)
class Neuron:
    number: int
    """MN number/index"""
    D_soma_meter: np.float64  # used for calculations, not for model
    """Soma Diameter [m]"""
    V_rest_mV: np.float64
    """Resting potential [mV]"""
    V_th_mV: np.float64
    V_reset_mV: np.float64
    V_init_mV: np.float64
    E_L_mV: np.float64
    # TODO Write in thesis how unrealistic these arbitrary linear distributions are.
    doublet_rheobase_coefficient: (
        np.float64
    )  # Rheobase coefficient for doublet threshold

    gain_leak: np.float64
    gain_exc: np.float64

    @property
    def D_soma(self) -> np.float64:
        """Soma diameter in [μm]"""
        return self.D_soma_meter * 1e6  # Soma diameter in [μm]

    @property
    def S_soma_meter(self) -> np.float64:
        """Soma Surface Area in [m^2]"""  # From Caillet table 4
        return self.D_soma_meter * 5.5e-3  # square meters [m^2].

    @property
    def S_soma(self) -> np.float64:
        """Soma Surface Area in micro_Meters ^2"""
        return (
            self.S_soma_meter * 1e12  # mAtHEmAtiCs need to square the micro also
        )  # Neuron surface area  [μ m^2]

    @property
    def R_Mohm(self) -> np.float64:
        """Membrane Resistance in Mega_Ohm"""  # From Caillet table 4
        R_unit = 1.68e-10 * np.pow(self.S_soma_meter, -2.43)  # Input resistance [Ω]
        return R_unit * 1e-6  # Input resistance [MΩ]

    @property
    def C_nF(self) -> np.float64:
        """Membrane capacitance nano_Farad"""  # From Caillet table 4
        C_unit = 7.9e-5 * self.D_soma_meter  # Membrane capacitance [F]
        return C_unit * 1e9  # Membrane capacitance [nF]

    @property
    def tau_ms(self) -> np.float64:
        """Membrane time constant milli_Seconds"""  # From Caillet table 4
        tau_unit = 2.3e-9 * np.pow(
            self.D_soma_meter, -1.48
        )  # 7.9e-5 * D_soma_unit * R_unit  # Membrane time constant [s]
        return tau_unit * 1e3  # Membrane time constant [ms]

    @property
    def I_rheo_ampere(self) -> np.float64:
        """Rheobase current Ampere"""  # From Caillet table 4
        return 7.8e2 * np.pow(self.D_soma_meter, 2.52)  # Rheobase current [A]

    @property
    def I_rheobase(self) -> np.float64:
        """Rheobase current nano_Ampere"""
        return self.I_rheo_ampere * 1e9  # Rheobase current [nA]

    @property
    def Rheobase_threshold(self) -> np.float64:
        """Doublet threshold (Rheobase current) nano_Ampere"""  # TODO From Paper..
        return (
            self.I_rheobase * self.doublet_rheobase_coefficient
        )  # Doublet current threshold [nA]

    @property
    def tref_seconds(self) -> np.float64:
        """Refractory time in Seconds"""  # From Caillet table 4
        # factor 0.3 -> decreased refractory time to make it work...
        return (
            0.3 * 0.2 * 2.7e-8 * np.pow(self.D_soma_meter, -1.51)
        )  # Refractory time [s]

    @property
    def tref(self) -> np.float64:
        """Refractory time in milli_Seconds"""
        return self.tref_seconds * 1e3  # Refractory time [ms]

    @property
    def g_L(self) -> np.float64:
        """Leak conductance  micro_Siemens"""
        # (g_L = C / tau or 1 / R)
        return 1 / self.R_Mohm * 10  # Leak Conductance [μS]


class NeuronFactory:

    @staticmethod
    def create_neuron(
        soma_diameter: np.float64,
        number: int,
        num_total_neurons: int,  # Needed for rheobase threshold coefficients
    ) -> Neuron:
        """Create parameters for a single neuron based on soma surface area"""
        leak_coefficient = np.linspace(0.25, 0.15, num_total_neurons)[
            number
        ]  # Leakage coefficient
        return Neuron(
            number=number,
            doublet_rheobase_coefficient=np.linspace(3.4, 2.1, num_total_neurons)[
                number
            ],  # Rheobase threshold coefficient
            gain_exc=leak_coefficient,
            gain_leak=leak_coefficient,
            D_soma_meter=soma_diameter,
            V_rest_mV=V_REST,  # Resting potential [mV]
            V_th_mV=V_THRESHOLD,  # Threshold potential [mV]
            V_reset_mV=V_RESET,  # Reset potential [mV]
            V_init_mV=V_INIT,  # Initial potential [mV]
            E_L_mV=E_LEAK,  # leak reversal potential [mV]
        )

    @staticmethod
    def create_neuron_pool(number_of_neurons: np.float64 = 300) -> List[Neuron]:
        """Create a neuron pool for a number of neurons"""

        soma_diameters = soma_diameter_vector(total_neurons=number_of_neurons)

        neuron_list = [
            NeuronFactory.create_neuron(
                soma_diameter=soma_diameter,
                number=i,
                num_total_neurons=number_of_neurons,
            )
            for i, soma_diameter in enumerate(soma_diameters)
        ]

        return neuron_list


def soma_diameter_vector(
    soma_Dmin_m=SOMA_DIAM_MIN_METER, soma_Dmax_m=SOMA_DIAM_MAX_METER, total_neurons=300
):
    """Generate vector of soma diameter [m] and soma surface areas [m^2]"""

    i_values = (
        np.linspace(0, total_neurons, total_neurons) / total_neurons
    )  # Normalized indices [0..1]
    # soma_diam_m_vec_1 = np.linspace(soma_Dmin_m, soma_Dmax_m, total_neurons)
    soma_diam_m_vec_2 = soma_Dmin_m + (i_values**2) * (soma_Dmax_m - soma_Dmin_m)

    return soma_diam_m_vec_2


if __name__ == "__main__":

    neurons = NeuronFactory.create_neuron_pool()
    print(neurons)
#     len = 300
#     d_vector, s_vector = soma_diameter_vector()

#     # a = Neuron(name="Kim", size=5)
#     # print(a)

#     # a.size = 30
#     # print(a)
