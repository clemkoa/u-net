import torch
from torch import nn
import torch.nn.functional as F


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up_scale = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

    def forward(self, x1, x2):
        print('before', x1.shape)
        x1 = self.up_scale(x1)
        print('after', x1.shape)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        print(x.shape)
        return x

class last_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.first_conv = double_conv(1, 64)
        self.first_pool = nn.MaxPool2d(2, stride=2, padding=0)
        self.second_conv = double_conv(64, 64)
        self.first_up = up(128, 64)
        self.third_conv = double_conv(128, 64)
        self.last_conv = nn.Conv2d(64, 2, 1, padding=0)

    def forward(self, x):
        x1 = self.first_conv(x)
        x1_pool = self.first_pool(x1)
        x2 = self.second_conv(x1_pool)
        x2_up = self.first_up(x2, x1)
        x3 = self.third_conv(x2_up)
        output = self.first_pool(x3)
        return output
