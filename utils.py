# utils.py

import json
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

    for _ in range(sample_size):
        x: int = random.randint(0, 1)
        y: int = random.randint(0, 1)

        sample: list[float] = [x, y]
        
        # XOR
        actual: list[float] = [int(not (x == y))]
        # AND
        # actual: list[float] = [int(x and y)]
        # OR
        # actual: list[float] = [int(x or y)]
        # NAND
        # actual: list[float] = [int(not (x and y))]
        # NXOR
        # actual: list[float] = [int(x == y)]


        samples.append(sample)
        actuals.append(actual)

    if data_path is not None:
        with open(data_path) as file:
            data = json.load(file)
        with open(data_path, 'w') as file:
            data["samples"] = samples
            data["actuals"] = actuals
            json.dump(data, file, indent=4)

    return (samples, actuals)


