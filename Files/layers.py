# layers.py

import json

from Files.neuron import Neuron


class Layers:
    _neurons_path: str = "Files/neurons.json"
    
    def __init__(self, config: dict, data: dict):
        self.config: dict = config
        self.data: dict = data
        layers: list[int] = self.config['layers']
        activations: list[str] = self.config['activations']
        weights: list[list[list[float]]] = self.data['weights']
        biases: list[list[float]] = self.data['biases']
        
        self.layers: list[list[Neuron]] = [
            [
                Neuron(
                    name=f"n_{l+1}_{n+1}",
                    weights=weights[l][n],
                    bias=biases[l][n],
                    activation=activations[l]
                )
                for n in range(layer)
            ]
            for l, layer in enumerate(layers)
        ]

    
    def save(self) -> None:
        data: dict = {}
        for layer in self.layers:
            for n in layer:
                data[n.name] = {
                    "weights": n.weights,
                    "bias": n.bias,
                    "score": n.score,
                    "activation": n.activation,
                    "slope": n.slope,
                    "delta": n.delta,
                    "input": n.input,
                    "output": n.output
                }
        with open(self._neurons_path, "w") as file:
            json.dump(data, file, indent=4)

    
    def __iter__(self):
        return iter(self.layers)


    def __len__(self):
        return len(self.layers)


    def __getitem__(self, index):
        return self.layers[index]