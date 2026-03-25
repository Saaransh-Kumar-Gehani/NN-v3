# utils.py

import random, math


def randominit(config: dict) -> None:
    seed: int | None = config.get('seed', None)
    sample_size: int = config.get('sample_size', 10)
    parameter_size: int = config.get('parameter_size') or (print("[WARNING] Parameter size is not given, defaulting to 4."), 4)[1]
    noise: float = config.get('noise', 0.0)

    random.seed(seed)

    samples: list[list[float]] = []
    actuals: list[float] = []

    for _ in range(sample_size):
        # RULE
        # score = 0.6*weather + 0.7*free_time + 0.2*money + 0.9*energy

        weather, free_time, money, energy, *_ = [random.uniform(-0.5, 0.5) for _ in range(parameter_size)]

        sample: list[float] = [weather, free_time, money, energy]

        energy = energy if energy > 0 else 1.5*energy
        free_time = 0.7*free_time if free_time > -0.1 else 1.5*free_time
        score: float = 0.6*weather + 0.7*free_time + 0.2*money + energy

        actual: float = 1 / (1 + math.exp(-5*score))

        samples.append(sample)
        actuals.append(actual)

