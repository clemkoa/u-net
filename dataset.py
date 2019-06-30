from PIL import Image, ImageSequence
import os
import numpy as np

def load_train_dataset(data_folder):
    path_train_images = os.path.join(data_folder, 'train-volume.tif')
    path_test_images = os.path.join(data_folder, 'test-volume.tif')
    path_train_labels = os.path.join(data_folder, 'train-labels.tif')

    images = [np.array([[np.array(page)]]) for page in ImageSequence.Iterator(Image.open(path_train_images))]
    labels = [np.array([[np.array(page)]]) for page in ImageSequence.Iterator(Image.open(path_train_labels))]
    return images, labels
