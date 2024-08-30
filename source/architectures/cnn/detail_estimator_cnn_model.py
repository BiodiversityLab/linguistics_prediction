import torch
from torch import nn
from utils.module_loader import load_module

import torchvision.models as models

class CNN(nn.Module):
    def __init__(self, input_channels, configuration):
        super(CNN, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)
        self.dropout = nn.Dropout(0.2)
        self.output_activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)
        x = self.output_activation(x)
        return x

def load_model(input_channels, configuration):
   combinedModel = CNN(input_channels, configuration)
   return combinedModel
