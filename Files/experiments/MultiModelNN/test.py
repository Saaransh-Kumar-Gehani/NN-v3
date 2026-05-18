# test.py

import math
import json, random, os

from Files.core.layers import Layers
from Files.core.trainer import Trainer
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

        
model: dict = {}
for d_i, model_path in enumerate(model_paths):
    with open(model_path) as model_file:
        try: model[d_i] = json.load(model_file)
        except json.decoder.JSONDecodeError: raise ValueError(f"Model [{model_path}] File is empty.")


counter = 0
L = []
T: list[Trainer] = []
for nn_i, nn in enumerate(config['NN']):
    for i in range(nn):
        layers = Layers(layers=config['layers'][nn_i][i], activations=config['activations'][nn_i][i], data=model[counter])
        L.append(layers)

        trainer = Trainer(layers=layers, activations=config['activations'][nn_i][i], lr=config['lr'], decay=config['decay'], softmax=config['softmax'][nn_i][i], loss=config['loss'][nn_i][i])
        T.append(trainer)

        counter += 1


samples, actuals = generate_dataset(parameter_size=config['parameter_size'], sample_size=50)

counter = 0
for sample, actual in zip(samples, actuals):
    print("Sample: ", sample)

    out1 = T[0].forward(sample=sample[0])
    out2 = T[1].forward(sample=sample[1])

    print("Output: ", out1, out2)
    # out1[0] = 1/(1 + math.exp(-out1[0]))
    # out2[0] = 1/(1 + math.exp(-out2[0]))
    # print("Output_sigmoid: ", out1, out2)

    sample = [out1[0], out2[0]]
    actual = actual_for_finals(out1, out2)

    final = T[2].forward(sample=sample)

    print("Final: ", final, "         Actual: ", actual)

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