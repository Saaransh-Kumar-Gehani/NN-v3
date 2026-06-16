# trainer.py

import math
import random
from layers import Layers
from neuron import Neuron


class Trainer:
    def __init__(self, Layers: Layers, lr: float = 0.01, decay: float = 0.01, loss: str = 'MSE', softmax: bool = False, *_, **__):
        self.Layers: Layers = Layers
        self.lr: float = lr
        self.decay: float = decay
        self.softmax: bool = softmax

        if loss.upper() in ['MSE', 'BCE', 'CCE']:
            self.loss: str = loss.upper()
        else:
            raise ValueError("<=> Loss Function [{loss}] is not supported.".format(loss))
        
        # {[WARNINGS]}
        if self.loss == 'BCE' and self.Layers[-1][0].activation != 'sigmoid':
            print("<=> [WARNING] BCE is used without sigmoid.")
        if self.loss == 'CCE' and not self.softmax:
            print("<=> [WARNING] CCE is used without softmax.")
        if self.softmax and self.Layers[-1][0].activation != 'linear':
            print("<=> [WARNING] Softmax is used without linear.")
        
    
    def train(self, samples: list[list[float]], actuals: list[list[float]]) -> float:
        for sample, actual in zip(samples, actuals):
            self.forward(sample=sample)

            self.backward(actual=actual)

        i = random.choice(range(len(samples)))
        loss: float = self.compute_loss(sample=samples[i], actual=actuals[i])

        return loss

        
    def forward(self, sample: list[float]) -> list[float]:
        out: list[float] = sample
        for layer in self.Layers:
            t: list[float] = []
            for neuron in layer:
                t.append(neuron.predict(input=out))
            out: list[float] = t

        if self.softmax:
            out = self.get_softmax(output=out)

            for n_i, neuron in enumerate(self.Layers[-1]):
                neuron.softmax = out[n_i]

        return out
    

    def backward(self, actual: list[float]) -> None:
        last_layer: list[Neuron] = self.Layers[-1]
        deltas: list[float] = self.compute_delta(last_layer=last_layer, actual=actual)

        for n_i, neuron in enumerate(last_layer):
            neuron.delta = deltas[n_i]

        for index in range(len(self.Layers)-1, 0, -1):
            next_layer: list[Neuron] = self.Layers[index]
            prev_layer: list[Neuron] = self.Layers[index-1]

            for prev_i, prev_neuron in enumerate(prev_layer):
                hidden_error: float = 0.0
                for next_neuron in next_layer:
                    hidden_error += next_neuron.delta * next_neuron.weight[prev_i]
                
                prev_neuron.delta = hidden_error * prev_neuron.slope

        # Updating
        for layer in self.Layers:
            for neuron in layer:
                for w in range(len(neuron.weight)):
                    neuron.weight[w] -= self.lr * (neuron.delta * neuron.input[w] + self.decay * neuron.weight[w])
                neuron.bias -= self.lr * (neuron.delta + self.decay * neuron.bias)

    
    def compute_delta(self, last_layer: list[Neuron], actual: list[float]) -> list[float]:
        deltas: list[float] = []
        match self.loss:
            case 'MSE':
                deltas: list[float] = [((neuron.output - act)*neuron.slope) for neuron, act in zip(last_layer, actual)]
            case 'BCE':
                deltas: list[float] = [((neuron.output - act)/(neuron.output*(1.0 - neuron.output))*neuron.slope) for neuron, act in zip(last_layer, actual)]
            case 'CCE':
                if self.softmax:
                    deltas: list[float] = [(-(act - neuron.softmax)) for neuron, act in zip(last_layer, actual)]
                else:
                    deltas: list[float] = [((-act/neuron.output)*neuron.slope) for neuron, act in zip(last_layer, actual)]
        
        return deltas
            

    def compute_loss(self, sample: list[float], actual: list[float]) -> float:
        out = self.forward(sample=sample)

        loss: float = 0.0
        for o, a in zip(out, actual):
            match self.loss:
                case 'MSE':
                    loss += (a - o)**2
                case 'BCE':
                    if o <= 0 or o >= 1:
                        print("<=> [WARNING] Invalid BCE input:", o)
                        continue
                    loss += -(a*math.log(o) + (1.0 - a)*math.log(1 - o))
                case 'CCE':
                    if o <= 0:
                        print("<=> [WARNING] Invalid CCE input:", o)
                        continue
                    loss += -(a*math.log(o))
        
        return loss
    

    def get_softmax(self, output: list[float]) -> list[float]:
        m = max(output)
        e = [math.exp(o - m) for o in output]
        s = sum(e)
        output: list[float] = [e[i]/s for i in range(len(e))]

        return output
                

        