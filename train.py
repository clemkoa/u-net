import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image

from dataset import load_train_dataset, load_test_dataset
from unet import UNet

data_folder = 'data'
model_path = 'model/unet.pt'

def train():
    model = UNet()
    optimizer = optim.SGD(model.parameters(), lr = 0.0003)
    for (input, label, weight) in load_train_dataset(data_folder):
        target = torch.from_numpy(label // 255).float()
        output = model(torch.from_numpy(input.astype(np.float32))).reshape((512, 512, 2))
        weight = torch.from_numpy(weight)
        weight = weight.repeat(2, 1).reshape(output.shape)

        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        print(loss)
        torch.save(model.state_dict(), model_path)
    return

if __name__ == "__main__":
    train()
