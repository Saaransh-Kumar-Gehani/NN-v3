# nn.py

import json
from layers import Layers
from trainer import Trainer
from utils import init, generate_dataset


class NN:
    samples = []

    def __init__(self, config_path: str, data_path: str):
        self.data_path: str = data_path
        open(config_path, "a").close()
        open(data_path, "a").close()

        with open(config_path) as file:
            self.config: dict = json.load(file)

        init(**self.config, data_path=data_path)

        with open(data_path) as file:
            data: dict = json.load(file)

        self.Layers: Layers = Layers(**self.config, data=data)
        self.Trainer: Trainer = Trainer(Layers=self.Layers, **self.config)

    
    def make_dataset(self) -> tuple[list[list[float]], list[list[float]]]:
        self.samples, self.actuals = generate_dataset(data_path=self.data_path, **self.config)

        return self.samples, self.actuals

    
    def train(self) -> None:
        if not self.samples:
            raise ValueError("<=> Dataset not initializated.")

        self.losses: list[float] = []
        for epoch in range(self.config['epoch_size']):
            loss: float = self.Trainer.train(samples=self.samples, actuals=self.actuals)
            self.losses.append(loss)

    
    def show_losses(self, loss_interval: int) -> list[float]:
        return self.losses[::loss_interval]

