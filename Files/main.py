# main.py

import json
import time

from Files.layers import Layers
from Files.trainer import Trainer
from Files.utils import init, generate_dataset


config_path: str = "Files/config.json"
data_path: str = "Files/data.json"
neurons_path: str = "Files/neurons.json"

# Creating all necessary files to prevent FileNotFoundError
open(config_path, "a").close()
open(data_path, "a").close()
open(neurons_path, "a").close()


with open(config_path) as config_file:
    try: config: dict = json.load(config_file)
    except json.decoder.JSONDecodeError: raise ValueError("Config File is empty.")

init(layers=config['layers'], seed=config['seed'], parameter_size=config['parameter_size'], data_path=data_path)
samples, actuals = generate_dataset(seed=config['seed'], parameter_size=config['parameter_size'], sample_size=config['sample_size'], noise=config['noise'], data_path=data_path)

with open(data_path) as data_file:
    try: data: dict = json.load(data_file)
    except json.decoder.JSONDecodeError: raise ValueError("Data File is empty.")


layers = Layers(config=config, data=data)

trainer = Trainer(config=config, layers=layers, loss=config['loss'])

t1 = time.time()
epoch_losses: list[float] = []
for epoch in range(config['epoch_size']):
    losses = trainer.train(samples=samples, actuals=actuals)

    epoch_losses.append(sum(losses)/config["sample_size"])
t2 = time.time()

layers.save()

print("Training Time: ", t2-t1)
print("Losses: ", [epoch_losses[i] for i in range(len(epoch_losses)) if (i+1)%10 == 0])

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