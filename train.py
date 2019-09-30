import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from dataset import load_train_dataset, load_test_dataset
from unet import UNet

data_folder = 'data'
model_path = 'model/unet.pt'

def train():
    model = UNet()
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    for (input, label) in load_train_dataset(data_folder):
        output = model(torch.from_numpy(input.astype(np.float32)))
        loss = F.binary_cross_entropy(output, torch.from_numpy(label // 255).float())

        loss.backward()
        optimizer.step()
        print(loss)
        torch.save(model.state_dict(), model_path)
    return

if __name__ == "__main__":
    train()
