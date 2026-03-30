# utils.py

import random, math, json


def init(config: dict, data_path: str) -> None:
    seed: int | None = config.get('seed', None)
    parameter_size: int = config.get('parameter_size') or (print("[WARNING] Parameter size is not given, defaulting to 4."), 4)[1]
    layers: list[int] = config['layers']

    random.seed(seed)

    with open(data_path, "w") as file:
        data = {
            "weights": [
                [
                    [
                        random.uniform(-0.5, 0.5)
                        for _ in range(parameter_size if l==0 else layers[l-1])
                    ]
                    for _ in range(neurons)
                ]
                for l, neurons in enumerate(layers)
            ],
            "biases": [
                [
                    random.uniform(-0.2, 0.2)
                    for _ in range(neurons)
                ]
                for neurons in layers
            ]
        }
                
        json.dump(data, file, indent=4)




def generate_dataset(config: dict, data_path: str) -> tuple[list[list[float]], list[float]]:
    seed: int | None = config.get('seed', None)
    sample_size: int = config.get('sample_size', 10)
    parameter_size: int = config.get('parameter_size') or (print("[WARNING] Parameter size is not given, defaulting to 4."), 4)[1]
    noise: float = config.get('noise', 0.0)

    random.seed(seed+1)

    samples: list[list[float]] = []
    actuals: list[float] = []

    for _ in range(sample_size):
        # RULE
        # score = 0.6*weather + 0.7*free_time + 0.2*money + 0.9*energy

        weather, free_time, money, energy, *_ = [random.uniform(-0.5, 0.5) for _ in range(parameter_size)]

        sample: list[float] = [weather, free_time, money, energy]

        energy_adj = 1.0*energy if energy > 0.0 else 1.5*energy
        free_time_adj = 0.8*free_time if free_time > -0.1 else 1.5*free_time
        weather_adj = 0.5*weather if weather > 0.0 else 0.8*weather
        score: float = weather_adj + free_time_adj + 0.1*money + energy_adj

        actual: float = 1 / (1 + math.exp(-5*score))

        samples.append(sample)
        actuals.append(actual)

    try: data: dict = json.load(open(data_path))
    except: data: dict = {}
    data["samples"] = samples
    data["actuals"] = actuals
    json.dump(data, open(data_path, "w"), indent=4)

    return samples, actuals