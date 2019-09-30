import os
import numpy as np
from PIL import Image
import PIL.ImageOps
import torch
import torch.optim as optim
import torch.nn.functional as F

from dataset import load_train_dataset, load_test_dataset
from unet import UNet

data_folder = 'data'
model_path = 'model/unet.pt'

def predict():
    model = UNet()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    for input in load_test_dataset(data_folder):
        output = model(torch.from_numpy(input.astype(np.float32))).detach().numpy()
        input_array = input.reshape((512, 512))
        output_array = output.argmax(1).reshape((512, 512)) * 255
        input_img = Image.fromarray(input_array)
        output_img = PIL.ImageOps.invert(Image.fromarray(output_array.astype(dtype=np.uint16)).convert('L'))
        input_img.show()
        output_img.show()
    return

if __name__ == "__main__":
    predict()
