'''
Part of code has been implemented by https://github.com/achaman2/truly_shift_invariant_cnns
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from architectures.APS_segmentation.aps.APSPool import ApsDown, ApsUp, get_pad_layer


class circular_pad(nn.Module):
    def __init__(self, padding = (1, 1, 1, 1)):
        super(circular_pad, self).__init__()
        self.pad_sizes = padding
        
    def forward(self, x):
            
        return F.pad(x, pad = self.pad_sizes , mode = 'circular')




class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, padding_mode = 'circular'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        if padding_mode == 'circular':
            
            self.double_conv = nn.Sequential(
                circular_pad((1, 1, 1, 1)),
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=0),
                nn.BatchNorm2d(mid_channels),
                nn.ELU(inplace=True),
                
                circular_pad((1, 1, 1, 1)),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.ELU(inplace=True)
            )

        
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ELU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ELU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, padding_mode = 'circular', filter_size = 3, aps_criterion = 'l2', device='cpu'):
        super().__init__()

        self.maxpool = nn.Sequential(
            get_pad_layer(padding_mode)((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size = 2, stride = 1),
            ApsDown(channels = in_channels, filt_size = filter_size, stride = 2, 
                apspool_criterion = aps_criterion, pad_type = padding_mode, device=device)
            )

        
        self.double_conv = DoubleConv(in_channels, out_channels, padding_mode = padding_mode)

    def forward(self, x):
        down_out, polyphase_comp = self.maxpool(x)
        out = self.double_conv(down_out)
        
        return out, polyphase_comp


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False, padding_mode = 'circular', filter_size = 3, device='cpu'):
        super().__init__()
        
        self.padding_mode = padding_mode

        if bilinear:
            raise Exception('Implementation with bilinear mode currently not supported.')
            
            
        else:
            
            #replace conv transpose 2d with APS_up+circular_conv with kernel size 2
                
            self.up = nn.Sequential(ApsUp(channels = in_channels, filt_size = filter_size, stride = 2, pad_type = padding_mode, device=device),
                                    circular_pad((0, 1, 0, 1)),
                                    nn.Conv2d(in_channels , in_channels // 2, kernel_size=2, stride = 1))
                                    
            self.conv = DoubleConv(in_channels, out_channels, padding_mode = padding_mode)
            


    def forward(self, x1, x2, polyphase_indices):
        
        x1 = self.up({'inp': x1, 'polyphase_indices': polyphase_indices})
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
        

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)