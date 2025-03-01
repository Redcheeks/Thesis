import numpy as np
import matplotlib.pyplot as plt

## Caillet et. al. 2022 Table 3
## https://doi.org/10.7554/eLife.76489
## Values extracted with help of Chatgpt but manually checked and corrected

# Caillet 2022 Github, some values are from here and are not rounded
# https://github.com/ArnaultCAILLET/Caillet-et-al-2022-PLOS_Comput_Biol/blob/main/MN_properties_relationships_MOD.py


import numpy as np
import matplotlib.pyplot as plt

# Number of motoneurons - TODO This may be some type of input to a generating function?
num_neurons = 300
# Generate normalized indices
i_values = np.linspace(0, num_neurons, num_neurons) / num_neurons

soma_Dmin = (
    50e-6  # minimum soma diameter in meter, - TODO input to a generating function?
)
soma_Dmax = (
    100e-6  # maximum soma diameter in meter, - TODO input to a generating function?
)


# OPTION 1: Generate a range of soma sizes (S) in [meters]
S = np.linspace(soma_Dmin, soma_Dmax, num_neurons)


# OPTION 2: Calculate soma diameters using a quadratic relationship
S2 = soma_Dmin + (i_values**2) * (soma_Dmax - soma_Dmin)


# Extracted empirical coefficients (ka) and exponents (a) from Table 3
params = {
    "ACV": {"ka": 4.0, "a": 0.7},
    "AHP_S": {"ka": 6.1e3, "a": -1.2},
    "AHP_ACV": {"ka": 1.5e4, "a": -1.4},
    "R_S": {"ka": 1.5e5, "a": -2.1},
    "R_ACV": {"ka": 6.3e5, "a": -2.3},
    "R_AHP": {"ka": 6.2e-1, "a": 1.1},
    "Ith_R": {"ka": 1.1e3, "a": -1.0},
    "Ith_ACV": {"ka": 3.2e-6, "a": 3.7},
    "Ith_AHP": {"ka": 2.5e4, "a": -1.7},
    "C_R": {"ka": 2.4e2, "a": -0.4},
    "C_Ith": {"ka": 2.9e1, "a": 0.3},
    "C_AHP": {"ka": 2.8e2, "a": -0.4},
    "C_ACV": {"ka": 2.5, "a": 0.8},
    "Tau_R": {"ka": 8.7, "a": 0.5},
    "Tau_AHP": {"ka": 2.2, "a": 0.8},
    "Tau_Ith": {"ka": 2.3e2, "a": -0.4},
    "Tau_ACV": {"ka": 1.2e4, "a": -1.3},
}


def compute_and_plot_all_param(S):

    # Compute properties based on relationships in table 3
    ACV = params["ACV"]["ka"] * S ** params["ACV"]["a"]
    AHP_S = params["AHP_S"]["ka"] * S ** params["AHP_S"]["a"]
    AHP_ACV = params["AHP_ACV"]["ka"] * ACV ** params["AHP_ACV"]["a"]
    R_S = params["R_S"]["ka"] * S ** params["R_S"]["a"]
    R_ACV = params["R_ACV"]["ka"] * ACV ** params["R_ACV"]["a"]
    R_AHP = params["R_AHP"]["ka"] * AHP_ACV ** params["R_AHP"]["a"]
    Ith_R = params["Ith_R"]["ka"] * R_S ** params["Ith_R"]["a"]
    Ith_ACV = params["Ith_ACV"]["ka"] * ACV ** params["Ith_ACV"]["a"]
    Ith_AHP = params["Ith_AHP"]["ka"] * AHP_ACV ** params["Ith_AHP"]["a"]
    C_R = params["C_R"]["ka"] * R_S ** params["C_R"]["a"]
    C_Ith = params["C_Ith"]["ka"] * Ith_R ** params["C_Ith"]["a"]
    C_AHP = params["C_AHP"]["ka"] * AHP_ACV ** params["C_AHP"]["a"]
    C_ACV = params["C_ACV"]["ka"] * ACV ** params["C_ACV"]["a"]
    Tau_R = params["Tau_R"]["ka"] * R_S ** params["Tau_R"]["a"]
    Tau_AHP = params["Tau_AHP"]["ka"] * AHP_ACV ** params["Tau_AHP"]["a"]
    Tau_Ith = params["Tau_Ith"]["ka"] * Ith_R ** params["Tau_Ith"]["a"]
    Tau_ACV = params["Tau_ACV"]["ka"] * ACV ** params["Tau_ACV"]["a"]

    # Table 4 relationships from https://github.com/ArnaultCAILLET/Caillet-et-al-2022-PLOS_Comput_Biol/blob/main/MN_properties_relationships_MOD.py

    def Ith_S_func(S):
        return 3.82e8 * np.pow(S, 2.52)  # I_th_S_tb4 = 3.8e8 * np.pow(S , 2.52)

    def S_Ith(Ith):
        return 3.96e-4 * np.pow(Ith, 0.396)  # S_I_tb4 = 4e-4 * np.pow(I_th, 0.40)

    def R_S_func(S, kR=1.68 * 10**-10):  # R_S_tb4 = 1.7e-10 / np.pow(S, 2.43)
        return kR / S**2.43

    def C_S_func(S, Cm_rec):
        return Cm_rec * S  # [F]

    def C_D_func(D_soma):
        return 7.9e-5 * D_soma  # [F]

    def tau_R_C_func(R, C):
        return R * C  # [s]

    def AHP_D_func(D_soma):
        return 2.7e-8 * np.pow(D_soma, -1.51)

    def tref_AHP_func(
        AHP,
    ):  # based on https://www.biorxiv.org/content/10.1101/2024.10.05.616762v1
        return 0.2 * AHP

    def Ith_distrib_func(
        MN, true_MN_pop
    ):  # Obtained from literature data -- from Caillet Github
        """
        Function returning the typical rheobase distribution across a MN pool,
        obtained from rats and cats experimental data
        Parameters
        ----------
        $ MN : location of the MN in the real MN population, integer

        Returns
        -------
        $ Ith : rheobase [A] of the MN, float
        """
        Ith = 3.85e-9 * np.pow(
            np.pow(9.1, ((MN / true_MN_pop))), 1.1831
        )  # 2.09*10**-9*16**((MN/400)**1.5725)
        return Ith

    # Plotting
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))

    axs[0, 0].plot(S, ACV, label="ACV vs S", color="orange")
    axs[0, 0].set_xlabel("Soma Size (S)")
    axs[0, 0].set_ylabel("Axonal Conduction Velocity (ACV)")

    axs[0, 1].plot(S, AHP_S, label="AHP vs S", color="blue")
    axs[0, 1].set_xlabel("Soma Size (S)")
    axs[0, 1].set_ylabel("Afterhyperpolarization Duration (AHP)")

    axs[0, 2].plot(ACV, AHP_ACV, label="AHP vs ACV", color="green")
    axs[0, 2].set_xlabel("Axonal Conduction Velocity (ACV)")
    axs[0, 2].set_ylabel("Afterhyperpolarization Duration (AHP)")

    axs[1, 0].plot(S, R_S, label="R vs S", color="red")
    axs[1, 0].set_xlabel("Soma Size (S)")
    axs[1, 0].set_ylabel("Input Resistance (R)")

    axs[1, 1].plot(ACV, R_ACV, label="R vs ACV", color="purple")
    axs[1, 1].set_xlabel("Axonal Conduction Velocity (ACV)")
    axs[1, 1].set_ylabel("Input Resistance (R)")

    axs[1, 2].plot(AHP_ACV, R_AHP, label="R vs AHP", color="brown")
    axs[1, 2].set_xlabel("Afterhyperpolarization Duration (AHP)")
    axs[1, 2].set_ylabel("Input Resistance (R)")

    axs[2, 0].plot(R_S, Ith_R, label="Ith vs R", color="magenta")
    axs[2, 0].set_xlabel("Input Resistance (R)")
    axs[2, 0].set_ylabel("Rheobase Current (Ith)")

    axs[2, 1].plot(ACV, Ith_ACV, label="Ith vs ACV", color="cyan")
    axs[2, 1].set_xlabel("Axonal Conduction Velocity (ACV)")
    axs[2, 1].set_ylabel("Rheobase Current (Ith)")

    axs[2, 2].plot(AHP_ACV, Ith_AHP, label="Ith vs AHP", color="black")
    axs[2, 2].set_xlabel("Afterhyperpolarization Duration (AHP)")
    axs[2, 2].set_ylabel("Rheobase Current (Ith)")

    for ax in axs.flat:
        ax.legend()
        ax.grid(True)

    plt.tight_layout()


compute_and_plot_all_param(S)
compute_and_plot_all_param(S2)
plt.show()
