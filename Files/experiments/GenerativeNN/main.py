# main.py

import json, random, time, os

from Files.core.layers import Layers
from Files.core.trainer import Trainer
from Files.core.utils import init
from Files.experiments.GenerativeNN.utils import generate_dataset


BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
config_path: str = os.path.join(BASE_DIR, "config.json")
data_path: str = os.path.join(BASE_DIR, "data.json")
neurons_path: str = os.path.join(BASE_DIR, "neurons.json")
model_path: str = os.path.join(BASE_DIR, "model.json")

# Creating all necessary files to prevent FileNotFoundError
open(config_path, "a").close()
open(data_path, "a").close()
open(neurons_path, "a").close()
open(model_path, "a").close()


with open(config_path) as config_file:
    try: config: dict = json.load(config_file)
    except json.decoder.JSONDecodeError: raise ValueError("Config File is empty.")

init(layers=config['layers'], seed=config['seed'], parameter_size=config['parameter_size'], data_path=data_path)
samples, actuals = generate_dataset(seed=config['seed'], parameter_size=config['parameter_size'], sample_size=config['sample_size'], noise=config['noise'], data_path=data_path)

with open(data_path) as data_file:
    try: data: dict = json.load(data_file)
    except json.decoder.JSONDecodeError: raise ValueError("Data File is empty.")


layers = Layers(layers=config['layers'], activations=config['activations'], data=data)
layers._neurons_path = neurons_path
layers._model_path = model_path

trainer = Trainer(layers=layers, activations=config['activations'], lr=config['lr'], decay=config['decay'], softmax=config['softmax'], loss=config['loss'])

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
for sample, actual in zip(*generate_dataset(parameter_size=config['parameter_size'], sample_size=10)):
    outs = trainer.forward(sample=sample)

    print("Sample: ", sample, "  ->  ", outs, " (Predicted)  |  ", actual, " (Actual)\n")
    r = random.random()
    s = 0
    for prob in outs:
        s+=prob
        if r<s:
            # print("Sample: ", sample, "  ->  ", [0, 1, 2, 3, 4][outs.index(prob)], " (Predicted)  |  ", [0, 1, 2, 3, 4][actual.index(max(actual))], " (Actual)")
            break
    # if not ((out>0.5) ^ (actual>0.5)):
    #     correct += 1
    # else:
    #     wrong += 1

# print("Correct: ", correct, "Wrong: ", wrong)