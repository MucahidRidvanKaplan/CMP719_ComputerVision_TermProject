import torch
import torch.nn as nn
import torch.nn.functional as F

from model.isnet_package import resnet as resnet
from DCNv2.TTOA import TOAA_Block
from model.isnet_package import GatedSpatialConv as gatedSpatialConvolution
#from torchinfo import summary

"""
This is the main class where the layers of the ISNet model are defined.
"""

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class TFD_Block(nn.Module):
    def __init__(self, inch, outch):
        super(TFD_Block, self).__init__()
        self.res1 = resnet.BasicBlock(inch, outch, stride=1, downsample=None)
        self.gate = gatedSpatialConvolution.GatedSpatialConv2d(inch, outch)

    def forward(self, x, f_x):
        u_0 = x
        u_1, delta_u_0 = self.res1(u_0)
        u_gate = self.gate(u_1, f_x)
        u_2 = u_gate + u_1 - 3 * delta_u_0
        return u_2


class ISNet(nn.Module):
    def __init__(self, layer_blocks, channels):
        super(ISNet, self).__init__()

        stem_width = int(channels[0])

        # STEM BLOCK - ENCODER PART
        self.stem_layer = nn.Sequential(
            nn.Conv2d(3, stem_width, 3, 2, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(stem_width, stem_width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(True),

            nn.Conv2d(stem_width, 2 * stem_width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(2 * stem_width),
            nn.ReLU(True),

            nn.MaxPool2d(3, 2, 1),
        )

        # RESIDUAL BLOCK 1 - ENCODER PART
        self.ResidualBlock1 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[0],
                                               in_channels=channels[1], out_channels=channels[2], stride=2)

        # RESIDUAL BLOCK 2 - ENCODER PART
        self.ResidualBlock2 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[1],
                                               in_channels=channels[2], out_channels=channels[3], stride=2)

        # DECONVOLUTION & RESIDUAL BLOCK 1 - DECODER PART
        self.Deconvolution2 = nn.ConvTranspose2d(channels[3], channels[2], 4, 2, 1)

        # TTOA_2 - DECODER PART
        self.TOAA_2 = TOAA_Block(channels[2], channels[2])

        # DECONVOLUTION & RESIDUAL BLOCK 2 - DECODER PART
        self.Deconvolution1 = nn.ConvTranspose2d(channels[2], channels[1], 4, 2, 1)

        # TOAA_1 - DECODER PART
        self.TOAA_1 = TOAA_Block(channels[1], channels[1])

        # EDGE PART
        self.UpSampling = nn.Conv2d(3, 64, 1)

        self.TFD1 = TFD_Block(64, 64)

        self.TFD2 = TFD_Block(64, 64)

        self.TFD3 = TFD_Block(64, 64)

        self.Conv1_TFD = nn.Conv2d(64, 1, 1)
        self.Conv2_TFD = nn.Conv2d(32, 1, 1)
        self.Conv3_TFD = nn.Conv2d(16, 1, 1)

        self.fuse = nn.Conv2d(64, 1, kernel_size=1, padding=0, bias=False)

        # SIGMOID
        self.SigmoidFunc = nn.Sigmoid()

        # FCN HEAD
        self.head = FCNHead(channels[1], 1)

    def forward(self, x1, x1_grad):
        _, _, hei, wid = x1.shape
        x1_size = x1.size()

        x2 = self.stem_layer(x1)
        x3 = self.ResidualBlock1(x2)
        x4 = self.ResidualBlock2(x3)
        #x2_size = x2.size()
        #x3_size = x3.size()
        #x4_size = x4.size()

        x5 = self.Deconvolution2(x4)
        #x5_size = x5.size()
        x6 = self.TOAA_2(x5, x3)
        #x6_size = x6.size()

        x7 = x6 + x3
        #x7_size = x7.size()

        x8 = self.Deconvolution1(x7)
        #x8_size = x8.size()
        x9 = self.TOAA_1(x8, x2)
        #x9_size = x8.size()

        x10 = x9 + x2
        #x10_size = x10.size()

        x4_ = F.interpolate(self.Conv1_TFD(x4), size=[hei, wid], mode='bilinear', align_corners=True)
        x7_ = F.interpolate(self.Conv2_TFD(x7), size=[hei, wid], mode='bilinear', align_corners=True)
        x10_ = F.interpolate(self.Conv3_TFD(x10), size=[hei, wid], mode='bilinear', align_corners=True)

        #x4_size = x4_.size()
        #x7_size = x7_.size()
        #x10_size = x10_.size()

        x11 = F.interpolate(x1_grad, size=[hei, wid], mode='bilinear', align_corners=True)
        #x11_size = x11.size()
        #x1_grad_size = x1_grad.size()
        x11 = self.UpSampling(x1_grad)
        #x11_size = x11.size()
        x12 = self.TFD1(x11, x4_)
        #x12_size = x12.size()
        x13 = self.TFD2(x12, x7_)
        #x13_size = x13.size()
        x14 = self.TFD3(x13, x10_)
        #x14_size = x14.size()
        x14 = self.fuse(x14)
        #x14_size = x14.size()
        x14 = F.interpolate(x14, x1_size[2:], mode='bilinear', align_corners=True)
        #x14_size = x14.size()
        edge_out = self.SigmoidFunc(x14)
        #edge_out_size = edge_out.size()

        x10 = F.interpolate(x10, size=[hei, wid], mode='bilinear')
        #x10_out_size = x10.size()

        total_output = edge_out * x10 + x10
        #total_output_size = total_output.size()

        prediction = self.head(total_output)

        final_out = F.interpolate(prediction, size=[hei, wid], mode='bilinear')

        return final_out, edge_out

    def _make_layer(self, block, block_num, in_channels, out_channels, stride):
        layers = [block(in_channels, out_channels, stride,downsample=True)
                  if stride != 1 or in_channels != out_channels else block(in_channels, out_channels)]
        for _ in range(1, block_num):
            layers.append(block(out_channels, out_channels, 1, downsample=False))
        return nn.Sequential(*layers)


class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inter_channels),

            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Conv2d(inter_channels, out_channels,     1, 1, 0)
        )

    def forward(self, x):
        return self.block(x)



