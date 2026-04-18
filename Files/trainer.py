# trainer.py

import math

from Files.layers import Layers
from Files.neuron import Neuron


class Trainer:
    def __init__(self, config: dict, data: dict, layers: Layers, loss: str = 'MSE'):
        self.config: dict = config
        self.data: dict = data
        self.layers: Layers | list[list[Neuron]] = layers
        
        if loss.upper() in ['MSE', 'BCE', 'BCEWL']:
            self.loss: str = loss.upper()
        else:
            raise ValueError(f"The loss function [{loss}] is not supported.")

    
    def train(self, samples: list[list[float]], actuals: list[float]):
        for sample, actual in zip(samples, actuals):
            self.forward(sample=sample)

            self.backprop(actual=actual)

    
    def forward(self, sample: list[float]) -> list[float]:
        out = sample
        for layer in self.layers:
            out = [n.predict(out) for n in layer]
        
        return out


    def backprop(self, actual: float):
        last_layer: list[Neuron] = self.layers[-1]
        for n_next in last_layer:
            n_next.delta = self.compute_delta(layer='output', n=n_next, actual=actual)
        
        for i in range(len(self.layers)-2, 0, -1):
            next_layer: list[Neuron] = self.layers[i]
            prev_layer: list[Neuron] = self.layers[i-1]
            
            for i_prev, n_prev in enumerate(prev_layer):
                delta: float = 0.0
                for n_next in next_layer:
                    delta += (n_next.delta * n_next.weights[i_prev])
                n_prev.delta = self.compute_delta(layer='hidden', n=n_prev, delta=delta)

        for layer in self.layers:
            for n in layer:
                for w in range(len(n.weights)):
                    n.weights[w] -= self.config['lr'] * (n.delta * n.input[w] + self.config['decay'] * n.weights[w])
                
                n.bias -= self.config['lr'] * (n.delta)


    def compute_delta(self, layer: str, n: Neuron, actual: float = 0.0, delta: float = 0.0) -> float:
        match self.loss:
            case 'MSE':
                if layer == 'output':
                    _delta: float = 2*(n.output - actual) * n.slope
                elif layer == 'hidden':
                    _delta: float = (delta) * n.slope
                else:
                    raise SyntaxError("Incorrect layer parameter in `compute_delta()`.")
                
            case 'BCE':
                if self.config['activations'][-1] != 'sigmoid':
                    print("<=> [BCE] is applied with non-sigmoid output layer.")
                if layer == 'output':
                    _delta: float = (n.output - actual)
                elif layer == 'hidden':
                    _delta: float = (delta) * n.slope
                else:
                    raise SyntaxError("Incorrect layer parameter in `compute_delta()`.")
                
            case 'BCEWL':
                if self.config['activations'][-1] != 'linear':
                    print("<=> [BCEWL] is applied with non-linear output layer.")
                if layer == 'output':
                    _delta: float = ((1/(1 + math.exp(-5*n.output))) - actual)
                elif layer == 'hidden':
                    _delta: float = (delta) * n.slope
                else:
                    raise SyntaxError("Incorrect layer parameter in `compute_delta()`.")

        return _delta
    

    def compute_losses(self, samples: list[list[float]], actuals: list[float]) -> list[float]:
        losses: list[float] = []
        for sample, actual in zip(samples, actuals):
            out = self.forward(sample=sample)[0]

            if self.loss == 'MSE':
                loss = (out - actual)**2
            elif self.loss == 'BCE':
                out = max(min(out, 1 - 1e-15), 1e-15)
                loss = -(actual * math.log(out) + (1 - actual) * math.log(1 - out))
            elif self.loss == 'BCEWL':
                loss = max(out, 0) - out*actual + math.log(1 + math.exp(-abs(out)))
            else:
                loss = 0.0

            losses.append(loss)
        
        return losses
            