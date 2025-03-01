from abc import ABC, abstractmethod
import numpy as np
from neuron import Neuron
from typing import Tuple


class TimestepSimulation(ABC):

    # method to run simulation for one neuron
    @staticmethod
    @abstractmethod
    def simulate_neuron(
        sim_time: np.float64, timestep: np.float64, neuron: Neuron, Iinj
    ) -> Tuple[np.array, np.array]:
        pass


# method to run simulation for group of neurons
def simulate_neuron_ensamble(self):
    pass
