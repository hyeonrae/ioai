#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict

import torch
import torch.nn as nn


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=64):
        super().__init__()

        features = init_features

        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")
        self.decoder1 = self._block(features * 2, features, name="dec1")
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)


    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.maxpool(enc1))
        enc3 = self.encoder3(self.maxpool(enc2))
        enc4 = self.encoder4(self.maxpool(enc3))

        bottleneck = self.bottleneck(self.maxpool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    def _block(self, in_channels, out_channels, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "_conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (
                        name + "_norm1",
                        nn.BatchNorm2d(num_features=out_channels)
                    ),
                    (
                        name + "_relu1",
                        nn.ReLU(inplace=True)
                    ),
                    (
                        name + "_conv2",
                        nn.Conv2d(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (
                        name + "_norm2",
                        nn.BatchNorm2d(num_features=out_channels)
                    ),
                    (
                        name + "_relu2",
                        nn.ReLU(inplace=True)
                    ),
                ]
            )
        )

def type_1() -> UNet:
    return UNet(
        in_channels=3,
        out_channels=1,
        init_features=64
    )

if __name__ == "__main__":
    from torchinfo import summary
    model = type_1()
    model.eval()

    print(model)
    summary(model, (1, 3, 256, 256))
    #print(dict(model.named_parameters()))
