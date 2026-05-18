# main.py

import math
import json, random, time, os

from Files.core.layers import Layers
from Files.core.trainer import Trainer
from Files.core.utils import init
from Files.experiments.MultiModelNN.utils import generate_dataset, actual_for_finals


BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
config_path: str = os.path.join(BASE_DIR, "config.json")
data_paths: list[str] = [os.path.join(BASE_DIR, f"data{i}.json") for i in range(1, 4)]
neurons_paths: list[str] = [os.path.join(BASE_DIR, f"neurons{i}.json") for i in range(1, 4)]
model_paths: list[str] = [os.path.join(BASE_DIR, f"model{i}.json") for i in range(1, 4)]

# Creating all necessary files to prevent FileNotFoundError
open(config_path, "a").close()
for data_path, neurons_path, model_path in zip(data_paths, neurons_paths, model_paths):
    open(data_path, "a").close()
    open(neurons_path, "a").close()
    open(model_path, "a").close()


with open(config_path) as config_file:
    try: config: dict = json.load(config_file)
    except json.decoder.JSONDecodeError: raise ValueError("Config File is empty.")


counter = 0
for nn_i, nn in enumerate(config['NN']):
    for i in range(nn):
        init(layers=config['layers'][nn_i][i], seed=config['seed'], parameter_size=config['parameter_size'], data_path=data_paths[counter])

        counter += 1

samples, actuals = generate_dataset(seed=config['seed'], parameter_size=config['parameter_size'], sample_size=config['sample_size'], noise=config['noise'], data_paths=data_paths)
        
data: dict = {}
for d_i, data_path in enumerate(data_paths):
    with open(data_path) as data_file:
        try: data[d_i] = json.load(data_file)
        except json.decoder.JSONDecodeError: raise ValueError(f"Data [{data_path}] File is empty.")

counter = 0
outputs = []
trainers: list[Trainer] = []
for nn_i, nn in enumerate(config['NN'][:-1]):
    for i in range(nn):
        layers = Layers(layers=config['layers'][nn_i][i], activations=config['activations'][nn_i][i], data=data[counter])
        layers._neurons_path = neurons_paths[counter]
        layers._model_path = model_paths[counter]

        trainer = Trainer(layers=layers, activations=config['activations'][nn_i][i], lr=config['lr'], decay=config['decay'], softmax=config['softmax'][nn_i][i], loss=config['loss'][nn_i][i])
        trainers.append(trainer)

        t1 = time.time()
        epoch_losses: list[float] = []
        for epoch in range(config['epoch_size']):
            losses = []
            for sample, actual in zip(samples, actuals):
                trainer.forward(sample=sample[i])
                losses.append(trainer.compute_loss(sample=sample[i], actual=actual[i]))
                trainer.backprop(actual=actual[i])

            epoch_losses.append(sum(losses)/config["sample_size"])
        t2 = time.time()

        layers.save()

        print("Training Time: ", t2-t1)
        print("Losses: ", [epoch_losses[i] for i in range(len(epoch_losses)) if (i+1)%10 == 0])

        counter += 1

    for sample in samples:
        t = []
        for i in range(nn):
            out = trainers[i].forward(sample=sample[i])
            # out[0] = 1/(1 + math.exp(-out[0]))
            t.append(out)
        outputs.append(t)
    

actuals_finals = [
    actual_for_finals(o1, o2)
    for o1, o2 in outputs
]

for nn_i, nn in enumerate(config['NN']):
    if (len(config['NN']) - nn_i) != 1:
        continue
    for i in range(nn):
        layers = Layers(layers=config['layers'][nn_i][i], activations=config['activations'][nn_i][i], data=data[counter])
        layers._neurons_path = neurons_paths[counter]
        layers._model_path = model_paths[counter]

        trainer = Trainer(layers=layers, activations=config['activations'][nn_i][i], lr=config['lr'], decay=config['decay'], softmax=config['softmax'][nn_i][i], loss=config['loss'][nn_i][i])

        t1 = time.time()
        epoch_losses: list[float] = []
        for epoch in range(config['epoch_size']):
            losses = []
            for [o1, o2], actual in zip(outputs, actuals_finals):
                if actual is None:
                    continue
                sample = [o1[0], o2[0]]
                trainer.forward(sample=sample)
                losses.append(trainer.compute_loss(sample=sample, actual=actual))
                trainer.backprop(actual=actual)

            epoch_losses.append(sum(losses)/config["sample_size"])
        t2 = time.time()

        layers.save()
        with open(data_paths[-1]) as file:
            try:
                data: dict = json.load(file)
            except:
                data: dict = {}
        data["samples"] = outputs
        data["actuals"] = actuals_finals
        with open(data_paths[-1], "w") as file:
            json.dump(data, file, indent=4)

        print("Training Time: ", t2-t1)
        print("Losses: ", [epoch_losses[i] for i in range(len(epoch_losses)) if (i+1)%10 == 0])

        counter += 1













# correct = 0
# wrong = 0
# for sample, actual in zip(*generate_dataset(parameter_size=config['parameter_size'], sample_size=10)):
#     outs = trainer.forward(sample=sample)

#     # print("Sample: ", sample, "  ->  ", outs, " (Predicted)  |  ", actual, " (Actual)")
#     r = random.random()
#     s = 0
#     for prob in outs:
#         s+=prob
#         if r<s:
#             print("Sample: ", sample, "  ->  ", [0, 1, 2, 3, 4][outs.index(prob)], " (Predicted)  |  ", [0, 1, 2, 3, 4][actual.index(max(actual))], " (Actual)")
#             break
#     # if not ((out>0.5) ^ (actual>0.5)):
#     #     correct += 1
#     # else:
#     #     wrong += 1

# # print("Correct: ", correct, "Wrong: ", wrong)