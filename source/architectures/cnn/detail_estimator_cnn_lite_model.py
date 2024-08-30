import torch
from torch import nn
from utils.module_loader import load_module
import pdb
    

class CNN(nn.Module):
    def __init__(self, input_channels):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 256, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 8, 3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(8)
       # self.bn3 = nn.BatchNorm2d(128)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.act = nn.ReLU()

        self.fc1 = nn.Linear(576, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()

    def forward(self, x):
        #pdb.set_trace()
        x = self.maxpool(self.act(self.bn1(self.conv1(x))))
        x = self.maxpool(self.maxpool(self.act(self.bn2(self.conv2(x)))))
        #x = self.maxpool(self.act(self.bn3(self.conv3(x))))
       # pdb.set_trace()
        #x = self.maxpool(self.maxpool(self.maxpool(self.act(self.bn1(self.conv1(x))))))
        #x = self.maxpool(self.maxpool(self.act(self.bn1(self.conv1(x)))))
        

        x = torch.flatten(x, 1)
        x = self.act(self.fc1(self.dropout(x)))
        x = self.fc2(self.dropout(x))
        output = self.softplus(x)
        #output = self.tanh(self.fc1(self.dropout(x)))

        return output


def load_model(input_channels, configuration):
   
   combinedModel = CNN(input_channels)
   return combinedModel

