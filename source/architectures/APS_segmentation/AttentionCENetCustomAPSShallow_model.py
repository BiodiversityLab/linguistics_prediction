from utils.module_loader import load_module
from architectures.APS_segmentation.AttentionBlock import *
from architectures.APS_segmentation.aps.UNetAPSParts import *
from torch import nn
import pdb

class AttentionCENetCustomAPS(nn.Module):

    def __init__(self, input_channels, configuration, device, dropout_prob = 0.05):
        #padding mode could be zeros or circular
        super(AttentionCENetCustomAPS, self).__init__()
        self.out_channels = 1

        self.conv1 = ConvBlock(input_channels, 512, dropout_prob=dropout_prob)
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.Up1 = ReluUpConv(512, 64, dropout_prob)
        self.Att1 = AttentionBlock(F_g=64, F_l=512, n_coefficients=32)
        self.UpConv1 = ReluConvBlock(576, 32, dropout_prob)
        self.OutConv =  nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x1_down = self.maxpool(x1)

        d1 = self.Up1(x1_down)
      #  pdb.set_trace()
        s1 = self.Att1(gate=d1, skip_connection=x1)
        d1 = torch.cat((s1, d1), dim=1)
        d1 = self.UpConv1(d1)
        OutConv = self.OutConv(d1)
        out = self.sigmoid(OutConv)
 
        return out

def load_model(input_channels, configuration, device):
   combinedModel = AttentionCENetCustomAPS(input_channels, configuration, device)
   return combinedModel


