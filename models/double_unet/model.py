import torch
import torch.nn as nn
import torch.nn.functional


class Squeeze_Excite(nn.Module):

    def __init__(self, channel, reduction):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class VGGBlock(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.SE = Squeeze_Excite(out_channels, 8)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.SE(out)

        return(out)


def output_block():
    Layer = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1)),
                          nn.Sigmoid())
    return Layer


class DUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = VGGBlock(3, 64, 64)
        self.conv2 = VGGBlock(64, 128, 128)
        self.conv3 = VGGBlock(128, 256, 256)
        self.conv4 = VGGBlock(256, 512, 512)
        self.conv5 = VGGBlock(512, 512, 512)

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.Vgg1 = VGGBlock(1024, 256, 256)
        self.Vgg2 = VGGBlock(512, 128, 128)
        self.Vgg3 = VGGBlock(256, 64, 64)
        self.Vgg4 = VGGBlock(128, 32, 32)

        self.out = output_block()

        self.conv11 = VGGBlock(6, 32, 32)
        self.conv12 = VGGBlock(32, 64, 64)
        self.conv13 = VGGBlock(64, 128, 128)
        self.conv14 = VGGBlock(128, 256, 256)

        self.Vgg5 = VGGBlock(1024, 256, 256)
        self.Vgg6 = VGGBlock(640, 128, 128)
        self.Vgg7 = VGGBlock(320, 64, 64)
        self.Vgg8 = VGGBlock(160, 32, 32)

        self.out1 = nn.Conv2d(
            in_channels=32, out_channels=3, kernel_size=(1, 1))

    def forward(self, x):
        x1 = self.conv1(x)

        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        x5 = self.conv5(self.pool(x4))

        x5 = self.up(x5)
        x5 = torch.cat([x5, x4], 1)
        x6 = self.Vgg1(x5)

        x6 = self.up(x6)
        x6 = torch.cat([x6, x3], 1)
        x7 = self.Vgg2(x6)

        x7 = self.up(x7)
        x7 = torch.cat([x7, x2], 1)
        x8 = self.Vgg3(x7)

        x8 = self.up(x8)
        x8 = torch.cat([x8, x1], 1)
        x9 = self.Vgg4(x8)

        output1 = self.out(x9)
        output1 = x*output1

        x = torch.cat([x, output1], 1)

        x11 = self.conv11(x)

        x12 = self.conv12(self.pool(x11))
        x13 = self.conv13(self.pool(x12))
        x14 = self.conv14(self.pool(x13))

        y = self.pool(x14)

        y = self.up(y)
        y = torch.cat([y, x14, x4], 1)
        y = self.Vgg5(y)

        y = self.up(y)
        y = torch.cat([y, x13, x3], 1)
        y = self.Vgg6(y)

        y = self.up(y)
        y = torch.cat([y, x12, x2], 1)
        y = self.Vgg7(y)

        y = self.up(y)
        y = torch.cat([y, x11, x1], 1)
        y = self.Vgg8(y)

        output2 = self.out1(y)

        return output2
