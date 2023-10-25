import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from pathlib import Path
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils, datasets

from dataset import CellDataset
from unet import UNet

data_folder = "data"
model_folder = Path("model")
model_folder.mkdir(exist_ok=True)
model_path = "model/unet-voc.pt"
saving_interval = 1
epoch_number = 10
shuffle_data_loader = False

transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(), transforms.Grayscale()])
dataset = datasets.VOCSegmentation(
    data_folder,
    year="2007",
    download=True,
    image_set="train",
    transform=transform,
    target_transform=transform,
)

def train():
    cell_dataset = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=shuffle_data_loader)

    model = UNet(dimensions=22)
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    optimizer = optim.RMSprop(
        model.parameters(), lr=0.0001, weight_decay=1e-8, momentum=0.9
    )
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epoch_number):
        print(f"Epoch {epoch}")
        for i, batch in enumerate(cell_dataset):
            input, target = batch
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target.squeeze().type(torch.LongTensor))
            # step_loss = loss.item()
            loss.backward()
            optimizer.step()

            # HACK to run on a macbook, use a small dataset of 10 batches
            if i > 10:
                break

        if (epoch + 1) % saving_interval == 0:
            print("Saving model")

        torch.save(model.state_dict(), model_path)
    torch.save(model.state_dict(), model_path)
    return


if __name__ == "__main__":
    train()
