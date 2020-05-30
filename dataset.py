from PIL import Image, ImageSequence
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as models
from torchvision import transforms, utils
import torch
import random
from skimage import io, exposure


def load_train_dataset(data_folder):
    path_train_images = os.path.join(data_folder, 'train-volume.tif')
    path_train_labels = os.path.join(data_folder, 'train-labels.tif')

    images = np.array([[[np.array(page)]] for page in ImageSequence.Iterator(Image.open(path_train_images))])
    raw_labels = np.array([np.array(page) // 255 for page in ImageSequence.Iterator(Image.open(path_train_labels))])
    labels = np.zeros((raw_labels.reshape(-1).size, 2))
    labels[np.arange(raw_labels.reshape(-1).size), raw_labels.reshape(-1)] = 1
    return zip(images, labels.reshape((30, 512, 512, 2)))


def load_test_dataset(data_folder):
    path_test_images = os.path.join(data_folder, 'test-volume.tif')

    images = np.array([[[np.array(page)]] for page in ImageSequence.Iterator(Image.open(path_test_images))])
    return images


class CellDataset(Dataset):
    def __init__(self, data_folder, transform):
        self._transform = transform
        self._data_folder = data_folder
        self.build_dataset()

    def build_dataset(self):
        self._path_train_images = os.path.join(self._data_folder, 'train-volume.tif')
        self._path_train_labels = os.path.join(self._data_folder, 'train-labels.tif')
        self._images = io.imread(self._path_train_images)
        self._labels = io.imread(self._path_train_labels)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self._images[idx]).float().unsqueeze(0).unsqueeze(0)
        label = torch.from_numpy(self._labels[idx] // 255).long()
        return image, label
