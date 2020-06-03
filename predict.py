import numpy as np
from PIL import Image
import torch

from unet import UNet
from dataset import CellDataset

data_folder = 'data'
model_path = 'model/unet.pt'


def predict():
    model = UNet()
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    cell_dataset = CellDataset(data_folder, eval=True)
    model.load_state_dict(checkpoint)
    model.eval()
    for i in range(len(cell_dataset)):
        input, _ = cell_dataset[i]
        output = model(input).permute(0, 2, 3, 1).squeeze().detach().numpy()
        input_array = input.squeeze().detach().numpy()
        output_array = output.argmax(2) * 255
        input_img = Image.fromarray(input_array)
        output_img = Image.fromarray(output_array.astype(dtype=np.uint16)).convert('L')
        input_img.show()
        output_img.show()
    return


if __name__ == "__main__":
    predict()
