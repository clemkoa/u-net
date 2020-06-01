import os
from torch.utils.data import Dataset
import torch
import random
from skimage import io


class CellDataset(Dataset):
    def __init__(self, data_folder, eval=False):
        self._data_folder = data_folder
        self._eval = eval
        self.build_dataset()

    def build_dataset(self):
        self._path_images = os.path.join(self._data_folder, 'train-volume.tif')
        self._path_labels = os.path.join(self._data_folder, 'train-labels.tif')
        if self._eval:
            self._path_images = os.path.join(self._data_folder, 'test-volume.tif')
            self._path_labels = self._path_images
        self._images = io.imread(self._path_images)
        self._labels = io.imread(self._path_labels)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self._images[idx])
        label = torch.from_numpy(self._labels[idx] // 255).long()
        if random.randint(0, 1):
            image = image.flip(0)
            label = label.flip(0)
        if random.randint(0, 1):
            image = image.flip(1)
            label = label.flip(1)
        image = image.float().unsqueeze(0).unsqueeze(0)
        return image, label
