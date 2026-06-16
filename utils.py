# utils.py

import json
import math
import random


def init(layers: list[int], parameter_size: int, data_path: str, seed: int | None = None, *_, **__) -> None:
    random.seed(seed)
    weights: list[list[list[float]]] = []
    biases: list[list[float]] = []

    for layer_i, layer in enumerate(layers):
        weight_l: list[list[float]] = []
        bias_l: list[float] = []

        for neuron in range(layer):
            weight: list[float] = [random.uniform(-0.5, 0.5) for _ in range(parameter_size if layer_i == 0 else layers[layer_i - 1])]
            bias: float = random.uniform(-0.5, 0.5)

            weight_l.append(weight)
            bias_l.append(bias)
        
        weights.append(weight_l)
        biases.append(bias_l)

    with open(data_path, 'w') as file:
        json.dump({
            "weights": weights,
            "biases": biases
        }, file, indent=4)


def generate_dataset(data_path: str | None = None, seed: int | None = None, sample_size: int = 10, *_, **__) -> tuple[list[list[float]], list[list[float]]]:
    random.seed(seed)
    samples: list[list[float]] = []
    actuals: list[list[float]] = []

    for i in range(sample_size):
        # x: int = random.uniform(0, 1)
        # y: int = random.uniform(0, 1)
        # z: int = random.randint(0, 1)

        
        # sample: list[float] = [x, y]

        # MAP = lambda x: [int(x<0.5), int(x>=0.5 and x<1), int(x>=1)]

        # sample: list[float] = [int(bit) for bit in format(i%256, '08b')]
        # actual: list[float] = [0]*9
        # actual[sample.count(1)] = 1


        centers = [0.0, 1.0, 2.0, 3.0, 4.0]

        x = 4*random.uniform(0, 1)   # [0,1]

        sample: list[float] = [x]

        actual: list[float] = []

        for mu in centers:
            actual.append(math.exp(-((x - mu)**2)/2))

        s = sum(actual)
        actual = [a/s for a in actual]


        # XOR
        # actual: list[float] = [int(not (x == y))]
        # AND
        # actual: list[float] = [int(x and y)]
        # OR
        # actual: list[float] = [int(x or y)]
        # NAND
        # actual: list[float] = [int(not (x and y))]
        # NXOR
        # actual: list[float] = [int(x == y)]
        # Parity
        # actual: list[float] = [1 if sample.count(1)%2 else 0]

        samples.append(sample)
        actuals.append(actual)


    if data_path is not None:
        with open(data_path) as file:
            data = json.load(file)
        with open(data_path, 'w') as file:
            try: data["samples"].append(samples)
            except: data["samples"] = samples
            try: data["actuals"].append(actuals)
            except: data["actuals"] = actuals
            
            json.dump(data, file, indent=4)

    return (samples, actuals)


