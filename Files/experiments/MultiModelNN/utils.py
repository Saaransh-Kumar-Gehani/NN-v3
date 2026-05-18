# utils.py

import random, math, json


def generate_dataset(
        seed: int | None = None,
        parameter_size: int | None = 4,
        sample_size: int = 10,
        noise: float = 0.0,
        data_paths: list[str] | None = None
        ) -> tuple[list[list[list[float]]], list[list[list[float]]]]:

    if parameter_size is None:
        print("[WARNING] Parameter size is not given, defaulting to 4.")
        parameter_size = 4

    random.seed(seed)

    samples: list[list[list[float]]] = []
    actuals: list[list[list[float]]] = []

    for _ in range(sample_size):
        # RULE
        # score = 0.6*weather + 0.7*free_time + 0.2*money + 0.9*energy

        weather, energy, *_ = [random.uniform(-0.2, 0.8) for _ in range(parameter_size)]
        money, free_time, *_ = [random.uniform(-0.2, 0.8) for _ in range(parameter_size)]

        energy_adj = 1.5*energy
        weather_adj = 0.8*weather
        free_time_adj = 1.5*free_time


        sample: list[list[float]] = [[weather, energy], [money, free_time]]

        score: float = weather_adj + energy_adj
        score1: float = 0.1*money + free_time_adj

        actual: list[list[float]] = [[score], [score1]]

        samples.append(sample)
        actuals.append(actual)


    if data_paths is not None:
        for d_i, data_path in enumerate(data_paths[:-1]):
            if data_path is not None:
                with open(data_path) as file:
                    try:
                        data: dict = json.load(file)
                    except:
                        data: dict = {}
                data["samples"] = [sample[d_i] for sample in samples]
                data["actuals"] = [actual[d_i] for actual in actuals]
                with open(data_path, "w") as file:
                    json.dump(data, file, indent=4)

    return samples, actuals


def actual_for_finals(out1, out2):
    a = out1[0]
    b = out2[0]
    T1 = 0.9
    T2 = 0.6

    if a <= T1 and b <= T2:
        return [1.0, 0.0, 0.0, 0.0]   # rest

    elif T1 <= a and b <= T2:
        return [0.0, 0.0, 0.0, 1.0]   # go out

    elif a <= T1 and T2 <= b:
        return [0.0, 1.0, 0.0, 0.0]   # light

    elif T1 <= a and T2 <= b:
        return [0.0, 0.0, 1.0, 0.0]   # deep work