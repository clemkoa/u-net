import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from dataset import load_train_dataset
from unet import UNet
data_folder = 'data'

train_images, train_labels = load_train_dataset(data_folder)

model = UNet()
for train_image in train_images:
    print(model(torch.from_numpy(train_image.astype(np.float32))))
