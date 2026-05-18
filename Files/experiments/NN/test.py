# test.py

import json, os

from Files.core.layers import Layers
from Files.core.trainer import Trainer
from Files.experiments.NN.utils import generate_dataset


BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
config_path: str = os.path.join(BASE_DIR, "config.json")
model_path: str = os.path.join(BASE_DIR, "model.json")

# Creating all necessary files to prevent FileNotFoundError
open(config_path, "a").close()
open(model_path, "a").close()


with open(config_path) as config_file:
    try: config: dict = json.load(config_file)
    except json.decoder.JSONDecodeError: raise ValueError("Config File is empty.")

with open(model_path) as model_file:
    try: model: dict = json.load(model_file)
    except json.decoder.JSONDecodeError: raise ValueError("Model File is empty.")


layers = Layers(layers=config['layers'], activations=config['activations'], data=model)

trainer = Trainer(layers=layers, activations=config['activations'], lr=config['lr'], decay=config['decay'], softmax=config['softmax'], loss=config['loss'])

correct = 0
wrong = 0
losses = []
for sample, actual in zip(*generate_dataset(parameter_size=config['parameter_size'], sample_size=10000, noise=0.5)):
    outs = trainer.forward(sample=sample)
    for out, act in zip(outs, actual):
        if not ((out>0.5) ^ (act>0.5)):
            correct += 1
        else:
            wrong += 1
        # print("Sample: ", sample, "  ->  ", out, " (Predicted)  |  ", act, " (Actual)")
    if (correct + wrong) % 1000 == 0:
        losses.append(trainer.compute_loss(sample=sample, actual=actual))

print("Losses: ", losses)
print("Correct: ", correct, "Wrong: ", wrong)