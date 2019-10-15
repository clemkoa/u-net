from PIL import Image, ImageSequence
import os
import numpy as np

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
