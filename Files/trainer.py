# trainer.py

import math

from Files.layers import Layers
from Files.neuron import Neuron


class Trainer:
    def __init__(self, config: dict, layers: Layers, loss: str = 'MSE'):
        self.config: dict = config
        self.layers: Layers | list[list[Neuron]] = layers
        
        if loss.upper() in ['MSE', 'BCE', 'BCEWL', 'CCE']:
            self.loss: str = loss.upper()
        else:
            raise ValueError(f"The loss function [{loss}] is not supported.")

    
    def train(self, samples: list[list[float]], actuals: list[list[float]]) -> list[float]:
        losses: list[float] = []
        for sample, actual in zip(samples, actuals):
            self.forward(sample=sample)

            losses.append(self.compute_loss(sample=sample, actual=actual))

            self.backprop(actual=actual)
        
        return losses

    
    def forward(self, sample: list[float]) -> list[float]:
        out = sample
        for layer in self.layers:
            out = [n.predict(out) for n in layer]

        if self.config['softmax']:
            max_logit = max(out)
            exps = [math.exp(z - max_logit) for z in out]
            sum_exps = sum(exps)
            out = [e / sum_exps for e in exps]
        for i, n in enumerate(self.layers[-1]):
            n.output = out[i]

        return out


    def backprop(self, actual: list[float]) -> None:
        last_layer: list[Neuron] = self.layers[-1]
        for n_i, n_next in enumerate(last_layer):
            n_next.delta = self.compute_delta(layer='output', n=n_next, actual=actual[n_i])
        
        for i in range(len(self.layers)-1, 0, -1):
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
                
            case 'CCE':
                if self.config['activations'][-1] != 'linear':
                    print("<=> [CCE] is applied with non-linear output layer.")
                if self.config['softmax'] is False:
                    print("<=> [CCE] is applied with non-softmax output layer.")
                if layer == 'output':
                    _delta: float = (n.output - actual)
                elif layer == 'hidden':
                    _delta: float = (delta) * n.slope
                else:
                    raise SyntaxError("Incorrect layer parameter in `compute_delta()`.")

        return _delta
    

    def compute_loss(self, sample: list[float], actual: list[float]) -> float:
        outs = self.forward(sample=sample)

        loss: float = 0.0
        for out, act in zip(outs, actual):
            if self.loss == 'MSE':
                loss += (out - act)**2
            elif self.loss == 'BCE':
                out = max(min(out, 1 - 1e-15), 1e-15)
                loss += -(act * math.log(out) + (1 - act) * math.log(1 - out))
            elif self.loss == 'BCEWL':
                loss += max(out, 0) - out*act + math.log(1 + math.exp(-abs(out)))
            elif self.loss == 'CCE':
                out = max(min(out, 1 - 1e-15), 1e-15)
                loss += -act * math.log(out)
        
        return loss
            