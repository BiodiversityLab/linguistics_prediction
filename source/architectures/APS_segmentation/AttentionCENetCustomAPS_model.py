from utils.module_loader import load_module
from architectures.APS_segmentation.AttentionBlock import *
from architectures.APS_segmentation.aps.UNetAPSParts import *
import torch
    

class AttentionCENetCustomAPS(nn.Module):
    #def __init__(self, input_numChannels, output_numChannels, dropout_prob = 0.0):
    def __init__(self, input_channels, configuration, device, dropout_prob = 0.0, inner_channels_list = [64, 128, 256, 512], filter_size = 3, bilinear=False, padding_mode = 'zeros'):
#         padding mode could be zeros or circular
        super(AttentionCENetCustomAPS, self).__init__()
        self.in_channels = input_channels
        self.out_channels = 1
        self.bilinear = bilinear
        self.padding_mode = padding_mode
        self.filter_size = filter_size

        
        self.inc = DoubleConv(input_channels, inner_channels_list[0], padding_mode = padding_mode)
        self.down1 = Down(inner_channels_list[0], inner_channels_list[1], padding_mode = padding_mode, filter_size = filter_size, aps_criterion =  'l2', device=device)
        self.down2 = Down(inner_channels_list[1], inner_channels_list[2], padding_mode = padding_mode, filter_size = filter_size, aps_criterion =  'l2', device=device)


        self.Conv1 = ReluConvBlock(input_channels, 32, dropout_prob)
        self.Conv2 = ReluConvBlock(256, 64, dropout_prob)
       

        self.Up3 = ReluUpConv(256, 128, dropout_prob)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv3 = ReluConvBlock(256, 64, dropout_prob)

        self.Up2 = ReluUpConv(64, 64, dropout_prob)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ReluConvBlock(128, 32, dropout_prob)

        self.Conv = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
     
        x2, poly2 = self.down1(x1)
        x3, poly3 = self.down2(x2)

        d3 = self.Up3(x3)
        
        s2 = self.Att3(gate=d3, skip_connection=x2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
    
        s1 = self.Att2(gate=d2, skip_connection=x1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv(d2)
        out = self.sigmoid(out)
        
 
        return out

def load_model(input_channels, configuration, device):
   combinedModel = AttentionCENetCustomAPS(input_channels, configuration, device)
   return combinedModel





