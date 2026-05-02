# layers.py

import json, os

from Files.core.neuron import Neuron


class Layers:
    _BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    _neurons_path: str = os.path.join(_BASE_DIR, "neurons.json")
    _model_path: str = os.path.join(_BASE_DIR, "model.json")
    
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
        weights: list[list[list[float]]] = []
        biases: list[list[float]] = []
        for l, layer in enumerate(self.layers):
            layer_w: list = []
            layer_b: list = []
            for n, neuron in enumerate(layer):
                layer_w.append(neuron.weights)
                layer_b.append(neuron.bias)
                data[neuron.name] = {
                    "weights": neuron.weights,
                    "bias": neuron.bias,
                    "score": neuron.score,
                    "activation": neuron.activation,
                    "slope": neuron.slope,
                    "delta": neuron.delta,
                    "input": neuron.input,
                    "output": neuron.output
                }
            weights.append(layer_w)
            biases.append(layer_b)
        data2: dict = {
            "weights": weights,
            "biases": biases
        }
        with open(self._neurons_path, "w") as file:
            json.dump(data, file, indent=4)
        with open(self._model_path, "w") as file:
            json.dump(data2, file, indent=4)

    
    def __iter__(self):
        return iter(self.layers)


    def __len__(self):
        return len(self.layers)


    def __getitem__(self, index):
        return self.layers[index]