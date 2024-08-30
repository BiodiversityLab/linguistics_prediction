'''
Part of code has been implemeted by https://github.com/sfczekalski/attention_unet
'''
import torch
from torch import nn
import numpy as np


class MaxPoolBlur(nn.Module):
    def __init__(self, channels):
        super(MaxPoolBlur, self).__init__()

        self.maxpoolBlur = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(channels),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        return self.maxpoolBlur(x)

class ReluConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super(ReluConvBlock, self).__init__()

        # number of input channels is a number of filters in the previous layer
        # number of output channels is a number of filters in the current layer
        # "same" convolutions
        if dropout_prob > 0.0:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.ELU(inplace=True),
                nn.Dropout2d(p=dropout_prob),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.ELU(inplace=True),
                nn.Dropout2d(p=dropout_prob)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.ELU(inplace=True),
                nn.Dropout2d(p=dropout_prob),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.ELU(inplace=True),
                nn.Dropout2d(p=dropout_prob)
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super(ConvBlock, self).__init__()

        # number of input channels is a number of filters in the previous layer
        # number of output channels is a number of filters in the current layer
        # "same" convolutions
        if dropout_prob > 0.0:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
                nn.Sigmoid(),
                nn.Dropout2d(p=dropout_prob),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_channels),
                #nn.ReLU(inplace=True)
                nn.Sigmoid(),
                nn.Dropout2d(p=dropout_prob),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
                nn.Sigmoid(),
            # nn.Dropout2d(p=dropout_prob),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_channels),
                #nn.ReLU(inplace=True)
                nn.Sigmoid(),
            #   nn.Dropout2d(p=dropout_prob),
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class ReluUpConv(nn.Module):

    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super(ReluUpConv, self).__init__()
        if dropout_prob > 0.0:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.ELU(inplace=True),
                nn.Dropout2d(p=dropout_prob),
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.ELU(inplace=True),
            #  nn.Dropout2d(p=dropout_prob),
            )

    def forward(self, x):
        x = self.up(x)
        return x
        

class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super(UpConv, self).__init__()
        if dropout_prob > 0.0:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_channels),
                #nn.ReLU(inplace=True)
                nn.Sigmoid(),
                nn.Dropout2d(p=dropout_prob),
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_channels),
                #nn.ReLU(inplace=True)
                nn.Sigmoid(),
            # nn.Dropout2d(p=dropout_prob),
            )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out