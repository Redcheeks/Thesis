from dataclasses import dataclass
import numpy as np

v_rest: np.float = (-70.0,)  # Resting potential in mV
v_th: np.float = (-50,)  # Threshold potential in mV
v_reset: np.float = (-70,)  # Reset potential in mV
v_init: np.float = (-70,)  # Initial potential in mV
e_l: np.float = (-70.0,)  # leak reversal potential in mV


@dataclass(frozen=True)
class Neuron:
    number: int
    D_soma: np.float
    S_soma: np.float
    R: np.float
    C: np.float
    tau_m: np.float
    I_th: np.float
    Doublet_current: np.float
    g_L: np.float
    V_rest: np.float
    V_th: np.float
    V_reset: np.float
    V_init: np.float
    E_L: np.float
    gain_leak: np.float
    gain_exc: np.float
    tref: np.float


class NeuronFactory:

    @staticmethod
    def create_neuron(
        soma_surface_area: np.float,
        number: int,
    ) -> Neuron:
        """Create parameters for a single neuron based on soma surface area"""

        return Neuron(
            number=i,
            D_soma=D_soma,
            S_soma=S_soma,
            R=R,  # Input resistance [MΩ]
            C=C,  # Membrane capacitance [nF]
            tau_m=tau,  # Membrane time constant [ms]
            I_th=I_th,  # Rheobase current [nA]
            Doublet_current=Doublet_Current,  # Double current thresholds [nA]
            g_L=g_L,  # Leak conductance [μS]
            V_rest=V_rest,  # Resting potential [mV]
            V_th=V_th,  # Threshold potential [mV]
            V_reset=V_reset,  # Reset potential [mV]
            V_init=V_init,  # Initial potential [mV]
            E_L=E_L,  # leak reversal potential [mV]
            gain_leak=gain_leak,
            gain_exc=gain_exc,
            tref=t_ref,  # Refractory period [ms])
        )


def soma_area_vector(soma_Dmin_m=50e-6, soma_Dmax_m=100e-6, len=300):
    """Generate vector of soma surface areas [m^2]"""
    soma_surface_area_m2 = (
        lambda d_m: d_m * 5.5e-3
    )  # (Neuron surface area) in square meters [m^2]. Caillet table 4
    i_values = np.linspace(0, len, len) / len  # Normalized indices [0..1]
    soma_diam_m_vec_1 = np.linspace(soma_Dmin_m, soma_Dmax_m, len)
    soma_diam_m_vec_2 = soma_Dmin_m + (i_values**2) * (soma_Dmax_m - soma_Dmin_m)

    return soma_surface_area_m2(soma_diam_m_vec_2)


if __name__ == "__main__":
    a = Neuron(name="Kim", size=5)
    print(a)

    a.size = 30
    print(a)
