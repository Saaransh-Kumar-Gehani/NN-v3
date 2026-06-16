# layers.py

from neuron import Neuron


class Layers:
    def __init__(self, layers: list[int], data: dict, activations: list[str], *_, **__):
        self.layers: list[list[Neuron]] = []
        weights: list[list[list[float]]] = data["weights"]
        biases: list[list[float]] = data["biases"]

        for layer_i, layer in enumerate(layers):
            listoflayer: list[Neuron] = []

            for neuron_i in range(layer):
                listoflayer.append(Neuron(
                    name=f"n_{layer_i+1}_{neuron_i+1}",
                    weight=weights[layer_i][neuron_i],
                    bias=biases[layer_i][neuron_i],
                    activation=activations[layer_i]
                ))

            self.layers.append(listoflayer)


    def __len__(self) -> int:
        return len(self.layers)
    

    def __add__(self, other) -> list:
        return self.layers + other.layers
    

    def __iter__(self):
        return iter(self.layers)
    

    def __getitem__(self, key) -> list[Neuron]:
        return self.layers[key]
    

    def __setitem__(self, key, value):
        raise ValueError("<=> Can not change Layers object.")
    


        