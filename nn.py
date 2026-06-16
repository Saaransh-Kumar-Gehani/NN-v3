# nn.py

import json
from neuron import Neuron
from layers import Layers
from trainer import Trainer
from utils import init, generate_dataset


class NN:
    samples: list[list[float]] = []
    actuals: list[list[float]] = []
    losses: list[float] = []

    def __init__(self, config_path: str, data_path: str, save_path: str):
        self.data_path: str = data_path
        self.save_path: str = save_path
        open(config_path, "a").close()
        open(data_path, "a").close()
        open(save_path+"/model.json", "a").close()
        open(save_path+"/neurons.json", "a").close()

        with open(config_path) as file:
            self.config: dict = json.load(file)

        init(**self.config, data_path=data_path)

        with open(data_path) as file:
            data: dict = json.load(file)

        self.Layers: Layers = Layers(**self.config, data=data)
        self.Trainer: Trainer = Trainer(Layers=self.Layers, **self.config)

    
    def train(self) -> None:
        samples, actuals = generate_dataset(data_path=self.data_path, **self.config)
        self.samples.append(samples)
        self.actuals.append(actuals)

        for epoch in range(self.config['epoch_size']):
            loss: float = self.Trainer.train(samples=samples, actuals=actuals)
            self.losses.append(loss)

    
    def show_losses(self, loss_interval: int) -> list[float]:
        return self.losses[::loss_interval]
    

    def save(self):
        data: dict = {}
        weights: list[list[list[float]]] = []
        biases: list[list[float]] = []
        neurons: list[Neuron] = []
        
        for layer in self.Layers:
            weight_l: list[list[float]] = []
            bias_l: list[float] = []
            for neuron in layer:
                weight_l.append(neuron.weight)
                bias_l.append(neuron.bias)
                neurons.append(neuron.__dict__)
            weights.append(weight_l)
            biases.append(bias_l)

        data['config'] = self.config
        data['weights'] = weights
        data['biases'] = biases

        with open(self.save_path+"/model.json", 'w') as file:
            json.dump(data, file, indent=4)

        with open(self.save_path+"/neurons.json", 'w') as file:
            json.dump({'neurons': neurons}, file, indent=4)

