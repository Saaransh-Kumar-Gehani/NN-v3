# main.py

import time
from nn import NN
from utils import generate_dataset


config_path: str = "NN-TESTING2/config.json"
data_path: str = "NN-TESTING2/data.json"


file = open("NN-TESTING2/output.txt", 'w')
for _ in range(1):
    nn1 = NN(config_path=config_path, data_path=data_path)
    t1 = time.time()
    nn1.train()
    t2 = time.time()


    file.write(f"\nTraining Time: {t2-t1}\n")
    file.write(str(nn1.show_losses(loss_interval=10))+"\n")

    samples, actuals = generate_dataset(sample_size=50)
    outs = [nn1.Trainer.forward(sample=sample) for sample in samples]

    for i in range(len(samples)):
        file.write(f"{samples[i]}  ->  {actuals[i]}  |  {outs[i]}\n")

file.close()