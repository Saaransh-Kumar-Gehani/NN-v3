# neuron.py

import math, json


class Neuron:
    score: float = 0.0
    output: float = 0.0
    slope: float = 0.0
    
    def __init__(self, name: str, weights: list[float] = [], bias: float = 0.0, activation: str = 'linear') -> None:
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
                output: float = 1/(1 + math.exp(-score))
                self.slope: float = output(1 - output)
            case 'relu':
                output: float = max(0, score)
                self.slope = math.ceil(min(1, max(0, score)))
            case 'tanh':
                output: float = math.tanh(score)
                self.slope: float = 1 - output**2


    def save(self):
        data = json.load(open("Files/neurons.json"))
        data[self.name] = {
            "weights": self.weights,
            "bias": self.bias,
            "score": self.score,
            "activation": self.activation,
            "slope": self.slope,
            "input": self.input,
            "output": self.output
        }