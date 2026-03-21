# layers.py

from Files.neuron import Neuron


class Layers:
    def __init__(self, config: dict, data: dict):
        self.config: dict = config
        layers: list[int] = self.config['layers']
        activations: list[str] = self.config['activations']
        
        self.layers: list[list[Neuron]] = [
            [
                Neuron(
                    name=f"n_{l+1}_{n+1}",
                    weights=data.get('weights', []),
                    bias=data.get('bias', 0.0),
                    activation=activations[l]
                )
                for n in range(l)
            ]
            for l in len(layers)
        ]

    
    def __iter__(self):
        return iter(self.layers)


    def __len__(self):
        return len(self.layers)


    def __getitem__(self, index):
        return self.layers[index]