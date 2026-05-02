# test.py

import json
import random

from Files.layers import Layers
from Files.trainer import Trainer
from Files.utils import generate_dataset


config_path: str = "Files/config.json"
model_path: str = "Files/model.json"

# Creating all necessary files to prevent FileNotFoundError
open(config_path, "a").close()
open(model_path, "a").close()


with open(config_path) as config_file:
    try: config: dict = json.load(config_file)
    except json.decoder.JSONDecodeError: raise ValueError("Config File is empty.")

with open(model_path) as model_file:
    try: model: dict = json.load(model_file)
    except json.decoder.JSONDecodeError: raise ValueError("Model File is empty.")


layers = Layers(config=config, data=model)

trainer = Trainer(config=config, layers=layers, loss=config['loss'])

correct = 0
wrong = 0
for sample, actual in zip(*generate_dataset(parameter_size=config['parameter_size'], sample_size=10)):
    outs = trainer.forward(sample=sample)

    # print("Sample: ", sample, "  ->  ", outs, " (Predicted)  |  ", actual, " (Actual)")
    r = random.random()
    s = 0
    for prob in outs:
        s+=prob
        if r<s:
            print("Sample: ", sample, "  ->  ", [0, 1, 2, 3, 4][outs.index(prob)], " (Predicted)  |  ", [0, 1, 2, 3, 4][actual.index(max(actual))], " (Actual)")
            break
    # if not ((out>0.5) ^ (actual>0.5)):
    #     correct += 1
    # else:
    #     wrong += 1

# print("Correct: ", correct, "Wrong: ", wrong)