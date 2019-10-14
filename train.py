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
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    for (input, label) in load_train_dataset(data_folder):
        optimizer.zero_grad()
        target = torch.from_numpy(label).float()
        output = model(torch.from_numpy(input.astype(np.float32))).permute(0, 2, 3, 1)
        loss = F.binary_cross_entropy(output, target)
        print(loss)

        loss.backward()
        optimizer.step()
        torch.save(model.state_dict(), model_path)
    return

if __name__ == "__main__":
    train()
