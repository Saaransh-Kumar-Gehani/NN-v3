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
for sample, actual in zip(*generate_dataset(parameter_size=4, sample_size=10000)):
    outs = trainer.forward(sample=sample)
    for out, act in zip(outs, actual):
        if not ((out>0.5) ^ (act>0.5)):
            correct += 1
        else:
            wrong += 1
        # print("Sample: ", sample, "  ->  ", out, " (Predicted)  |  ", act, " (Actual)")

print("Correct: ", correct, "Wrong: ", wrong)