
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two convolution layers with ReLU activation."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    """Small U-Net for binary brain tumour segmentation."""

    def __init__(self, in_channels=1, out_channels=1, features=16):
        super().__init__()

        self.encoder1 = DoubleConv(in_channels, features)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = DoubleConv(features, features * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(features * 2, features * 4)

        self.up2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = DoubleConv(features * 4, features * 2)

        self.up1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = DoubleConv(features * 2, features)

        self.output = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))

        dec2 = self.up2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.output(dec1)
