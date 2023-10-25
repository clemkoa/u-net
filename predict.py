import numpy as np
from PIL import Image
import torch

from unet import UNet
from dataset import CellDataset
from torchvision import transforms, utils, datasets

data_folder = "data"
model_path = "model/unet-voc.pt"

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


def predict():
    model = UNet(dimensions=22)
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    cell_dataset = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=shuffle_data_loader)
    model.load_state_dict(checkpoint)
    model.eval()
    for i, batch in enumerate(cell_dataset):
        input, _ = batch
        print(input.shape)
        # output = model(input).permute(0, 2, 3, 1).squeeze().detach().numpy()
        output = model(input).squeeze().detach().numpy()
        input_array = input.squeeze().detach().numpy()
        output_array = output[3]
        print(output_array.shape)
        print(output.max())
        print(input_array)
        input_img = Image.fromarray(input_array * 255)
        print(output_array.astype(dtype=np.uint16)* 10)
        input_img.show()
        for i in range(22):
            output_array = output[i]
            output_img = Image.fromarray(output_array.astype(dtype=np.uint16)).convert("L")
            output_img.show()
        break
    return


if __name__ == "__main__":
    predict()
