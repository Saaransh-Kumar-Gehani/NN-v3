# utils.py

import random, math, json


def init(
        layers: list[int],
        seed: int | None = None,
        parameter_size: int | None = 4,
        data_path: str = "Files/data.json"
        ) -> None:
    
    if parameter_size is None:
        print("[WARNING] Parameter size is not given, defaulting to 4.")
        parameter_size = 4

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




def generate_dataset(
        seed: int | None = None,
        parameter_size: int | None = 4,
        sample_size: int = 10,
        noise: float = 0.0,
        data_path: str | None = None
        ) -> tuple[list[list[float]], list[list[float]]]:

    if parameter_size is None:
        print("[WARNING] Parameter size is not given, defaulting to 4.")
        parameter_size = 4

    random.seed(seed)

    samples: list[list[float]] = []
    actuals: list[float] = []

    for _ in range(sample_size):
        # RULE

        num, *_ = [random.random() for _ in range(parameter_size)]

        sample: list[float] = [num]
        
        # i = int(num * 5)
        # actual = [(1.0 if j == i else 0.0) for j in range(5)]

        center = num * 5 + 0.5

        actual = []
        for j in range(5):
            val = math.exp(-(j - center)**2 / 2)   # gaussian
            actual.append(val)

        # normalize
        s = sum(actual)
        actual = [v/s for v in actual]

        samples.append(sample)
        actuals.append(actual)

    if data_path is not None:
        with open(data_path) as file:
            try:
                data: dict = json.load(file)
            except:
                data: dict = {}
        data["samples"] = samples
        data["actuals"] = actuals
        with open(data_path, "w") as file:
            json.dump(data, file, indent=4)

    return samples, actuals