# utils.py

import random, json


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
    actuals: list[list[float]] = []

    for _ in range(sample_size):
        # RULE
        # score = 0.6*weather + 0.7*free_time + 0.2*money + 0.9*energy

        weather, free_time, money, energy, *_ = [random.uniform(-0.5, 0.5) for _ in range(parameter_size)]

        sample: list[float] = [weather, free_time, money, energy]

        energy_adj = 1.0*energy if energy > 0.0 else 1.5*energy
        free_time_adj = 0.8*free_time if free_time > 0.0 else 1.5*free_time
        weather_adj = 0.5*weather if weather > 0.0 else 0.8*weather
        score: float = weather_adj + free_time_adj + 0.1*money + energy_adj + random.uniform(-noise, noise)

        # actual: float = 1 / (1 + math.exp(-score))

#         score = (
#     0.6 * weather +
#     0.7 * free_time +
#     0.2 * money +
#     0.9 * energy
# )

        actual: list[float] = [max(0.0, min(1.0, score))]

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