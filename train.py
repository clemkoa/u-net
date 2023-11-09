import os
import torch
import torch.optim as optim
from pathlib import Path
from torch import nn
from torchvision import transforms, utils, datasets

from unet import UNet

data_folder = "data"
model_folder = Path("model")
model_folder.mkdir(exist_ok=True)
model_path = "model/unet-voc.pt"
saving_interval = 10
epoch_number = 100
shuffle_data_loader = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(), transforms.Grayscale()])
dataset = datasets.VOCSegmentation(
    data_folder,
    year="2007",
    download=True,
    image_set="train",
    transform=transform,
    target_transform=transform,
)

def train():
    cell_dataset = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=shuffle_data_loader)

    model = UNet(dimensions=22)
    model.to(device)
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    optimizer = optim.RMSprop(
        model.parameters(), lr=0.0001, weight_decay=1e-8, momentum=0.9
    )
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epoch_number):
        print(f"Epoch {epoch}")
        losses = []
        for i, batch in enumerate(cell_dataset):
            input, target = batch
            input = input.to(device)
            target = target.type(torch.LongTensor).to(device)
            # HACK to skip the last item that has a batch size of 1, not working with the cross entropy implementation
            if input.shape[0] < 2:
                continue
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target.squeeze())
            # step_loss = loss.item()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        # print the average loss for that epoch.
        print(sum(losses) /len(losses))
        if (epoch + 1) % saving_interval == 0:
            print("Saving model")

        torch.save(model.state_dict(), model_path)
    torch.save(model.state_dict(), model_path)
    return


if __name__ == "__main__":
    train()
