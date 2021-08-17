import torch
import torch.nn as nn
import torch.nn.functional as F

from .hardnet_68 import hardnet
from .attn import AdditiveAttnGate


class NeoUNet(nn.Module):
    def __init__(self, pretrained=True, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

        self.encoder = hardnet(pretrained=pretrained)
        self.d0, self.d1, self.d2, self.d3, self.d4 = 64, 128, 320, 640, 1024

        self.decode0 = self._decoder_block(self.d0 * 2, self.d0)
        self.decode1 = self._decoder_block(self.d1 * 2, self.d1)
        self.decode2 = self._decoder_block(self.d2 * 2, self.d2)
        self.decode3 = self._decoder_block(self.d3 * 2, self.d3)

        self.out0 = self._out_block(self.d0)
        self.out1 = self._out_block(self.d1)
        self.out2 = self._out_block(self.d2)
        self.out3 = self._out_block(self.d3)

        self.attn_mid = AdditiveAttnGate(self.d3, self.d4)
        self.attn_3 = AdditiveAttnGate(self.d2, self.d3)
        self.attn_2 = AdditiveAttnGate(self.d1, self.d2)

        self.upsample_mid = self._upsampler_block(self.d4, self.d3)
        self.upsample_3 = self._upsampler_block(self.d3, self.d2)
        self.upsample_2 = self._upsampler_block(self.d2, self.d1)
        self.upsample_1 = self._upsampler_block(self.d1, self.d0)

    def _out_block(self, in_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, self.num_classes, kernel_size=1),
        )

    @property
    def output_scales(self):
        return 1., 1., 1., 1.

    def set_num_classes(self, num_classes: int):
        self.num_classes = num_classes

        self.out3 = self._out_block(self.d3)
        self.out2 = self._out_block(self.d2)
        self.out1 = self._out_block(self.d1)
        self.out0 = self._out_block(self.d0)

    def _decoder_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def _upsampler_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=4, stride=2,
            padding=1, bias=False
        )

    def forward(self, x):
        """Forward method

        :param inputs: Input tensor of size (B x C x W x H)
        :return: List of output for each level
        """

        o0, o1, o2, o3, o4 = self.encoder.forward_2(x)

        attn_mid = self.attn_mid(o3, o4)
        up_mid = self.upsample_mid(o4)

        merged_3 = torch.cat((up_mid, attn_mid), dim=1)
        decode_3 = self.decode3(merged_3)
        attn_3 = self.attn_3(o2, decode_3)
        out_3 = self.out3(decode_3)
        up_3 = self.upsample_3(decode_3)

        merged_2 = torch.cat((attn_3, up_3), dim=1)
        decode_2 = self.decode2(merged_2)
        attn_2 = self.attn_2(o1, decode_2)
        out_2 = self.out2(decode_2)
        up_2 = self.upsample_2(decode_2)

        merged_1 = torch.cat((attn_2, up_2), dim=1)
        decode_1 = self.decode1(merged_1)
        out_1 = self.out1(decode_1)
        up_1 = self.upsample_1(decode_1)

        merged_0 = torch.cat((o0, up_1), dim=1)
        decode_0 = self.decode0(merged_0)
        out_0 = self.out0(decode_0)

        output_size = x.size()[2:]
        out_0 = F.interpolate(out_0, size=output_size, mode='bilinear', align_corners=False)
        out_1 = F.interpolate(out_1, size=output_size, mode="bilinear")
        out_2 = F.interpolate(out_2, size=output_size, mode="bilinear")
        out_3 = F.interpolate(out_3, size=output_size, mode="bilinear")

        return out_0, out_1, out_2, out_3
