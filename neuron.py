from dataclasses import dataclass
import numpy as np

V_REST: np.float = -70.0  # Resting potential in mV
V_THRESHOLD: np.float = -50  # Threshold potential in mV
V_RESET: np.float = -70  # Reset potential in mV
V_INIT: np.float = -70  # Initial potential in mV
E_LEAK: np.float = -70.0  # leak reversal potential in mV

SOMA_DIAM_MIN_METER: np.float = 50e-6
SOMA_DIAM_MAX_METER: np.float = 100e-6


@dataclass(frozen=True)
class Neuron:
    number: int
    """MN number/index"""
    D_soma_meter: np.float  # used for calculations, not for model
    """Soma Diameter [m]"""
    V_rest_mV: np.float
    """Resting potential [mV]"""
    V_th_mV: np.float
    V_reset_mV: np.float
    V_init_mV: np.float
    E_L_mV: np.float
    # TODO Write in thesis how unrealistic these arbitrary linear distributions are.
    doublet_rheobase_coefficient: np.float  # Rheobase coefficient for doublet threshold
    leak_array_coefficient: np.float  # Leakage coefficient
    gain_leak: np.float = leak_array_coefficient
    gain_exc: np.float = leak_array_coefficient

    @property
    def D_soma(self) -> np.float:
        """Soma diameter in [μm]"""
        return self.D_soma_unit * 1e6  # Soma diameter in [μm]

    @property
    def S_soma_meter(self) -> np.float:
        """Soma Surface Area in [m^2]"""  # From Caillet table 4
        return self.D_soma_meter * 5.5e-3  # square meters [m^2].

    @property
    def S_soma(self) -> np.float:
        """Soma Surface Area in micro_Meters ^2"""
        return (
            self.S_soma_meter * 1e12  # mAtHEmAtiCs need to square the micro also
        )  # Neuron surface area  [μ m^2]

    @property
    def R_Mohm(self) -> np.float:
        """Membrane Resistance in Mega_Ohm"""  # From Caillet table 4
        R_unit = 1.68e-10 * np.pow(self.S_soma_meter, -2.43)  # Input resistance [Ω]
        return R_unit * 1e-6  # Input resistance [MΩ]

    @property
    def C_nF(self) -> np.float:
        """Membrane capacitance nano_Farad"""  # From Caillet table 4
        C_unit = 7.9e-5 * self.D_soma_meter  # Membrane capacitance [F]
        return C_unit * 1e9  # Membrane capacitance [nF]

    @property
    def tau_ms(self) -> np.float:
        """Membrane time constant milli_Seconds"""  # From Caillet table 4
        tau_unit = 2.3e-9 * np.pow(
            self.D_soma_meter, -1.48
        )  # 7.9e-5 * D_soma_unit * R_unit  # Membrane time constant [s]
        return tau_unit * 1e3  # Membrane time constant [ms]

    @property
    def I_rheo_ampere(self) -> np.float:
        """Rheobase current Ampere"""  # From Caillet table 4
        return 7.8e2 * np.pow(self.D_soma_meter, 2.52)  # Rheobase current [A]

    @property
    def I_rheobase(self) -> np.float:
        """Rheobase current nano_Ampere"""
        return self.I_rheo_ampere * 1e9  # Rheobase current [nA]

    @property
    def Rheobase_threshold(self) -> np.float:
        """Doublet threshold (Rheobase current) nano_Ampere"""  # TODO From Paper..
        return (
            self.I_rheobase * self.doublet_rheobase_coefficient
        )  # Doublet current threshold [nA]

    @property
    def tref_seconds(self) -> np.float:
        """Refractory time in Seconds"""  # From Caillet table 4
        # factor 0.3 -> decreased refractory time to make it work...
        return (
            0.3 * 0.2 * 2.7e-8 * np.pow(self.D_soma_meter, -1.51)
        )  # Refractory time [s]

    @property
    def tref(self) -> np.float:
        """Refractory time in milli_Seconds"""
        return self.tref_seconds * 1e3  # Refractory time [ms]

    @property
    def g_L(self) -> np.float:
        """Leak conductance  micro_Siemens"""
        # (g_L = C / tau or 1 / R)
        return 1 / self.R_Mohm  # Leak Conductance [μS]


class NeuronFactory:

    @staticmethod
    def create_neuron(
        soma_surface_area: np.float,
        soma_diameter: np.float,  # TODO We can probably remove this or surface area, only need one!!
        number: int,
        num_total_neurons: int,  # Needed for rheobase threshold coefficients
    ) -> Neuron:
        """Create parameters for a single neuron based on soma surface area"""

        return Neuron(
            number=number,
            S_soma_meter=soma_surface_area,
            doublet_rheobase_coefficient=np.linspace(3.4, 2.1, num_total_neurons)[
                number
            ],  # Rheobase threshold coefficient
            leak_array_coefficient=np.linspace(0.25, 0.15, num_total_neurons)[
                number
            ],  # Leakage coefficient
            D_soma_meter=soma_diameter,
            V_rest_mV=V_REST,  # Resting potential [mV]
            V_th_mV=V_THRESHOLD,  # Threshold potential [mV]
            V_reset_mV=V_RESET,  # Reset potential [mV]
            V_init_mV=V_INIT,  # Initial potential [mV]
            E_L_mV=E_LEAK,  # leak reversal potential [mV]
        )


def soma_area_vector(
    soma_Dmin_m=SOMA_DIAM_MIN_METER, soma_Dmax_m=SOMA_DIAM_MAX_METER, len=300
):
    """Generate vector of soma diameter [m] and soma surface areas [m^2]"""

    i_values = np.linspace(0, len, len) / len  # Normalized indices [0..1]
    soma_diam_m_vec_1 = np.linspace(soma_Dmin_m, soma_Dmax_m, len)
    soma_diam_m_vec_2 = soma_Dmin_m + (i_values**2) * (soma_Dmax_m - soma_Dmin_m)

    return soma_diam_m_vec_2, soma_surface_area_m2(
        soma_diam_m_vec_2
    )  # TODO We only need one but undecided which one!


if __name__ == "__main__":

    len = 300
    d_vector, s_vector = soma_area_vector()

    a = Neuron(name="Kim", size=5)
    print(a)

    a.size = 30
    print(a)
