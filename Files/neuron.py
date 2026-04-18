# neuron.py

import math, json


class Neuron:
    score: float = 0.0
    output: float = 0.0
    slope: float = 0.0
    delta: float = 0.0
    _neurons_path: str = "Files/neurons.json"
    
    def __init__(self, name: str, weights: list[float] = [], bias: float = 0.0, activation: str = 'linear'):
        self.name: str = name
        self.weights: list[float] = weights
        self.bias: float = bias

        if activation.lower() in ['linear', 'sigmoid', 'relu', 'tanh']:
            self.activation: str = activation.lower()
        else:
            raise ValueError(f"The activation function [{activation}] is not supported.")

    
    def predict(self, sample: list[float]) -> float:
        self.input: list[float] = sample
        self.score: float = sum(w*i for w, i in zip(self.weights, self.input))
        self.output: float = self.activate(self.score)

        return self.output
    

    def activate(self, score: float) -> float:
        match self.activation:
            case 'linear':
                output: float = score
                self.slope: float = 1.0
            case 'sigmoid':
                output: float = 1/(1 + math.exp(-5*score))
                self.slope: float = output*(1 - output)
            case 'relu':
                output: float = max(0, score)
                self.slope = math.ceil(min(1, max(0, score)))
            case 'tanh':
                output: float = math.tanh(score)
                self.slope: float = 1 - output**2
                
        return output


    def save(self) -> None:
        with open(self.neurons_path) as file:
            try:
                data: dict = json.load(file)
            except:
                data: dict = {}
        data[self.name] = {
            "weights": self.weights,
            "bias": self.bias,
            "score": self.score,
            "activation": self.activation,
            "slope": self.slope,
            "input": self.input,
            "output": self.output
        }
        with open(self.neurons_path, "w") as file:
            json.dump(data, file, indent=4)


    def __setattr__(self, name, value):
        if name == "_neurons_path":
            raise AttributeError("Cannot modify _neurons_path from instance.")
        super().__setattr__(name, value)


    @property
    def neurons_path(self) -> str:
        return Neuron._neurons_path