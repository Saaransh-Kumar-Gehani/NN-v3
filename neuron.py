# neuron.py

import math


class Neuron:
    score: float = 0.0
    slope: float = 0.0
    output: float = 0.0
    delta: float = 0.0

    def __init__ (self, name: str, weight: list[float], bias: float, activation: str = 'linear'):
        self.name: str = name
        self.weight: list[float] = weight
        self.bias: float = bias

        if activation.lower() in ['linear', 'sigmoid', 'tanh', 'relu', 'lrelu']:
            self.activation: str = activation.lower()
        else:
            raise ValueError("<=> Activation Function [{activation}] is not supported.".format(activation))
        
        
    def predict(self, input: list[float]) -> float:
        self.input: list[float] = input
        self.score: float = sum(w*i for w, i in zip(self.weight, self.input)) + self.bias

        o, s = self.activate(self.score)

        self.output: float = o
        self.slope: float = s

        return self.output

    
    def activate(self, score: float) -> tuple[float, float]:
        match self.activation:
            case 'linear':
                output = score
                slope = 1.0
            case 'sigmoid':
                output = 1.0/(1.0 + math.exp(-score))
                slope = output*(1.0 - output)
            case 'tanh':
                output = (1.0 - math.exp(-2*score))/(1.0 + math.exp(-2*score))
                slope = 1.0 - output**2
            case 'relu':
                output = score if score > 0 else 0.0
                slope = 1.0 if score > 0 else 0.0
            case 'lrelu':
                output = score if score > 0 else 0.01*score
                slope = 1.0 if score > 0 else 0.01
        
        return (output, slope)



