import torch
import torch.nn as nn
import torch.nn.functional as F
from dcn_v2 import DCN



class TOAA_Block(nn.Module):
    def __init__(self, low_channels, high_channels, c_kernel=3, r_kernel=3, use_att=False, use_process=True):
        super(TOAA_Block, self).__init__()

        if low_channels != high_channels:
            raise ValueError('Low and High channels need to be the same!')

        self.low_channels = low_channels
        self.high_channels = high_channels
        self.c_kernel = c_kernel
        self.r_kernel = r_kernel
        self.use_att = use_att
        self.use_process = use_process

        if self.use_process:
            self.preprocess = nn.Sequential(
                nn.Conv2d(self.low_channels, self.high_channels // 2, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(self.high_channels // 2, self.low_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.preprocess = None

        self.dcn_row = nn.Conv2d(self.low_channels, self.high_channels, kernel_size=(1, self.r_kernel),
                                 stride=1, padding=(0, self.r_kernel // 2))
        self.dcn_column = nn.Conv2d(self.low_channels, self.high_channels, kernel_size=(self.c_kernel, 1),
                                    stride=1, padding=(self.c_kernel // 2, 0))
        self.sigmoid = nn.Sigmoid()

        if self.use_att:
            self.csa = nn.Conv2d(self.low_channels, self.high_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.csa = None

    def forward(self, a_low, a_high):
        if self.preprocess is not None:
            a_low = self.preprocess(a_low)
            a_high = self.preprocess(a_high)

        a_low_column = self.dcn_column(a_low)
        a_low_column_weight = self.sigmoid(a_low_column)
        a_low_column_weighted = a_low_column_weight * a_high
        a_column_output = a_low + a_low_column_weighted

        a_low_row = self.dcn_row(a_low)
        a_low_row_weight = self.sigmoid(a_low_row)
        a_low_row_weighted = a_low_row_weight * a_high
        a_row_output = a_low + a_low_row_weighted

        if self.csa is not None:
            a_output = self.csa(a_column_output + a_row_output)
        else:
            a_output = a_column_output + a_row_output

        return a_output
