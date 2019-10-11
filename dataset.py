from PIL import Image, ImageSequence
import os
import numpy as np

def load_train_dataset(data_folder):
    path_train_images = os.path.join(data_folder, 'train-volume.tif')
    path_train_labels = os.path.join(data_folder, 'train-labels.tif')

    images = np.array([[[np.array(page)]] for page in ImageSequence.Iterator(Image.open(path_train_images))])
    raw_labels = np.array([np.array(page) // 255 for page in ImageSequence.Iterator(Image.open(path_train_labels))])
    means = np.expand_dims(np.expand_dims(np.mean(raw_labels, axis=(1, 2)), 1), 1)
    weights = np.multiply(np.ones(raw_labels.shape), means)
    weights = np.multiply(weights, 10.0 / means, where=raw_labels < 0.0001)
    labels = np.zeros((raw_labels.reshape(-1).size, 2))
    labels[np.arange(raw_labels.reshape(-1).size), raw_labels.reshape(-1)] = 1
    return zip(images, labels.reshape((30, 512, 512, 2)), weights.reshape((30, 512, 512)))

def load_test_dataset(data_folder):
    path_test_images = os.path.join(data_folder, 'test-volume.tif')

    images = np.array([[[np.array(page)]] for page in ImageSequence.Iterator(Image.open(path_test_images))])
    return images
