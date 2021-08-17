import torch


class Upsample(torch.torch.nn.Module):
    """ torch.nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="bilinear"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x


class BasicConv(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, bn=True, relu=True):
        super(BasicConv, self).__init__()
        
        self.conv = torch.nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, 
                              stride=stride,
                              padding=padding, 
                              dilation=dilation, 
                              bias=False)
        self.bn = torch.nn.BatchNorm2d(out_planes) if bn else None
        self.relu = torch.nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class RFBblock(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFBblock, self).__init__()
        self.branch0 = torch.nn.Sequential(
            BasicConv(in_channel, out_channel, 1, stride=1, padding=0),
        )
        self.branch1 = torch.nn.Sequential(
            BasicConv(in_channel, out_channel, 1, stride=1, padding=0, bn=False, relu=False),
            BasicConv(out_channel, out_channel, 3, stride=1, padding=1, bn=False, relu=False)
        )
        self.branch2 = torch.nn.Sequential(
            BasicConv(in_channel, out_channel, 1, stride=1, padding=0, bn=False, relu=False),
            BasicConv(out_channel, out_channel, 3, stride=1, padding=1, bn=False, relu=False),
            BasicConv(out_channel, out_channel, 3, stride=1, dilation=3, padding=3, bn=False, relu=False)
        )
        self.branch3 = torch.nn.Sequential(
            BasicConv(in_channel, out_channel, 1, stride=1, padding=0, bn=False, relu=False),
            BasicConv(out_channel, out_channel, 5, stride=1, padding=2, bn=False, relu=False),
            BasicConv(out_channel, out_channel, 3, stride=1, dilation=5, padding=5, bn=False, relu=False),
        )
        self.conv_cat = BasicConv(4*out_channel, out_channel, 1, padding=0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        return out


class Head(torch.nn.Module):
    def __init__(self, channel, n_classes):

        super(Head, self).__init__()
        
        self.head = BasicConv(channel, n_classes, 3, padding=1, bn=False, relu=False)

    def forward(self, x):
        out = self.head(x)

        return out



class LSC(torch.nn.Module):
    def __init__(self, channel):

        super(LSC, self).__init__()

        self.upsample_1 = torch.nn.Sequential(
            Upsample(scale_factor=2),
        )

        self.upsample_3 = torch.nn.Sequential(
            Upsample(scale_factor=2),
        )

        self.block_22 = torch.nn.Sequential(
            BasicConv(channel*2, channel, 3, padding=1),
            BasicConv(channel, channel, 3, padding=1)
        )

        self.block_33 = torch.nn.Sequential(
            BasicConv(channel*2, channel, 3, padding=1),
            BasicConv(channel, channel, 3, padding=1)
        )
        
    def forward(self, x1, x2, x3):
        x_22 = torch.cat((self.upsample_1(x1), x2), 1)
        x_22 = self.block_22(x_22)

        x_33 = torch.cat((self.upsample_3(x_22), x3), 1)
        x_33 = self.block_33(x_33)

        return x_33


class IDA(torch.nn.Module):
    def __init__(self, channel):

        super(IDA, self).__init__()

        self.upsample_1 = torch.nn.Sequential(
            Upsample(scale_factor=2),
        )

        self.upsample_2 = torch.nn.Sequential(
            Upsample(scale_factor=2),
        )

        self.upsample_3 = torch.nn.Sequential(
            Upsample(scale_factor=2),
        )

        self.block_22 = torch.nn.Sequential(
            BasicConv(channel*2, channel, 3, padding=1),
            BasicConv(channel, channel, 3, padding=1)
        )
        self.block_23 = torch.nn.Sequential(
            BasicConv(channel*2, channel, 3, padding=1),
            BasicConv(channel, channel, 3, padding=1)
        )
        self.block_33 = torch.nn.Sequential(
            BasicConv(channel*2, channel, 3, padding=1),
            BasicConv(channel, channel, 3, padding=1)
        )

        
    def forward(self, x1, x2, x3):
        x_22 = torch.cat((self.upsample_1(x1), x2), 1)
        x_22 = self.block_22(x_22)

        x_23 = torch.cat((self.upsample_2(x2), x3), 1)
        x_23 = self.block_23(x_23)

        x_33 = torch.cat((self.upsample_3(x_22), x_23), 1)
        x_33 = self.block_33(x_33)

        return x_33


class DIA(torch.nn.Module):
    def __init__(self, channel):

        super(DIA, self).__init__()

        self.upsample_1 = torch.nn.Sequential(
            Upsample(scale_factor=2),
        )

        self.upsample_2 = torch.nn.Sequential(
            Upsample(scale_factor=2),
        )

        self.upsample_3 = torch.nn.Sequential(
            Upsample(scale_factor=2),
        )

        self.block_22 = torch.nn.Sequential(
            BasicConv(channel*2, channel, 3, padding=1),
            BasicConv(channel, channel, 3, padding=1)
        )
        self.block_23 = torch.nn.Sequential(
            BasicConv(channel*2, channel, 3, padding=1),
            BasicConv(channel, channel, 3, padding=1)
        )
        self.block_33 = torch.nn.Sequential(
            BasicConv(channel*3, channel, 3, padding=1),
            BasicConv(channel, channel, 3, padding=1)
        )
        
    def forward(self, x1, x2, x3):
        x_22 = torch.cat((self.upsample_1(x1), x2), 1)
        x_22 = self.block_22(x_22)

        x_23 = torch.cat((self.upsample_2(x2), x3), 1)
        x_23 = self.block_23(x_23)

        x_33 = torch.cat((self.upsample_3(x_22), x_23, x3), 1)
        x_33 = self.block_33(x_33)

        return x_33



class DHA(torch.nn.Module):
    def __init__(self, channel):

        super(DHA, self).__init__()

        self.upsample_1 = torch.nn.Sequential(
            Upsample(scale_factor=2),
        )

        self.upsample_2 = torch.nn.Sequential(
            Upsample(scale_factor=2),
        )

        self.upsample_3 = torch.nn.Sequential(
            Upsample(scale_factor=2),
        )

        self.upsample_4 = torch.nn.Sequential(
            Upsample(scale_factor=2),
        )

        self.upsample_5 = torch.nn.Sequential(
            Upsample(scale_factor=4),
        )

        self.block_22 = torch.nn.Sequential(
            BasicConv(channel*2, channel, 3, padding=1),
            BasicConv(channel, channel, 3, padding=1)
        )
        self.block_23 = torch.nn.Sequential(
            BasicConv(channel*2, channel, 3, padding=1),
            BasicConv(channel, channel, 3, padding=1)
        )
        self.block_33 = torch.nn.Sequential(
            BasicConv(channel*3, channel, 3, padding=1),
            BasicConv(channel, channel, 3, padding=1)
        )
        self.block_32 = torch.nn.Sequential(
            BasicConv(channel*2, channel, 3, padding=1),
            BasicConv(channel, channel, 3, padding=1)
        )
        self.block_31 = torch.nn.Sequential(
            BasicConv(channel*2, channel, 3, padding=1),
            BasicConv(channel, channel, 3, padding=1)
        )

    def forward(self, x1, x2, x3):
        x_22 = torch.cat((self.upsample_1(x1), x2), 1)
        x_22 = self.block_22(x_22)

        x_23 = torch.cat((self.upsample_2(x2), x3), 1)
        x_23 = self.block_23(x_23)

        x_33 = torch.cat((self.upsample_3(x_22), x_23, x3), 1)
        x_33 = self.block_33(x_33)

        x_32 = torch.cat((self.upsample_4(x_22), x_33), 1)
        x_32 = self.block_32(x_32)

        x_31 = torch.cat((self.upsample_5(x1), x_32), 1)
        x_31 = self.block_31(x_31)

        return x_31

