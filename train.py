import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import random
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as models
from torchvision import transforms, utils
import torch

from dataset import CellDataset
from unet import UNet

data_folder = 'data'
model_path = 'model/unet.pt'
saving_interval = 50
epoch_number = 200


def train():
    cell_dataset = CellDataset(data_folder)
    model = UNet(dimensions=2)
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
    optimizer = optim.RMSprop(model.parameters(), lr=0.0001, weight_decay=1e-8, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epoch_number):
        i = random.randint(0, len(cell_dataset) - 1)
        input, target = cell_dataset[i]
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target.unsqueeze(0))
        step_loss = loss.item()
        print(f'Epoch: {epoch} \tLoss: {step_loss}')

        loss.backward()
        optimizer.step()

        if (epoch + 1) % saving_interval == 0:
            print('Saving model')

            torch.save(model.state_dict(), model_path)
    torch.save(model.state_dict(), model_path)
    return


if __name__ == "__main__":
    train()
