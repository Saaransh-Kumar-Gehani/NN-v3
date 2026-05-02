# utils.py

import random, json, os


def init(
        layers: list[int],
        seed: int | None = None,
        parameter_size: int | None = 4,
        data_path: str | None = None
        ) -> None:
    
    if data_path is None:
        BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
        data_path: str = os.path.join(BASE_DIR, "data.json")

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

