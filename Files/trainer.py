# trainer.py

from Files.layers import Layers
from Files.neuron import Neuron


class Trainer:
    def __init__(self, config: dict, data: dict, layers: Layers):
        self.config: dict = config
        self.data: dict = data
        self.layers: Layers = layers

    
    def train(self, samples: list[float], actuals)