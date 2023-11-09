import numpy as np
from PIL import Image
import torch

from unet import UNet
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
        output = model(input).detach()
        input_array = input.squeeze().detach().numpy()
        output_array = output.argmax(dim=1)
        # Simple conversion to black and white.
        # Everything class 0 is background, make everything else white.
        # This is bad for images with several classes.
        output_array = torch.where(output_array > 0, 255, 0)
        input_img = Image.fromarray(input_array * 255)
        input_img.show()
        output_img = Image.fromarray(output_array.squeeze().numpy().astype(dtype=np.uint16)).convert("L")
        output_img.show()
        # Just showing first ten images. Change as you wish!
        if i > 10:
            break
    return


if __name__ == "__main__":
    predict()
