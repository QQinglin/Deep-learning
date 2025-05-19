import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout


class ResBlock(torch.nn.Module):
    def __init__(self,in_channels, out_channels, stride):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride =stride, padding=(1,1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=(1,1))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Sequential()
        self.do1 = Dropout(p=0.3)

        self.r2 = nn.LeakyReLU()

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(nn.BatchNorm2d(in_channels), nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                                                              stride=stride))

    def forward(self,x):
        out = x.clone()
        # First sequence
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.do1(out)

        # Second sequence
        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection: Add the input(after adjusting with 1x1 Conv) to the output
        skip = self.skip(x)
        out = out + skip

        # Final ReLU after adding the skip connection
        out = self.r2(out)
        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        #initialize layers
        self.Conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2)
        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.do = Dropout(p=0.3)

        self.block1 = ResBlock(in_channels=64, out_channels=64, stride=1)
        self.block2 = ResBlock(in_channels=64, out_channels=128, stride=2)
        self.block3 = ResBlock(in_channels=128, out_channels=256, stride=2)
        self.block4 = ResBlock(in_channels=256, out_channels=512, stride=2)
        ##
        self.block5 = ResBlock(in_channels=512, out_channels=512, stride=1)
        ##
        self.global_avg_pool1 = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        ##
        self.fc = nn.Linear(in_features=512, out_features=2)
        self.d = Dropout(p=0.3)
        ##
        self.fc2 = nn.Linear(256, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        # initial layers
        x = self.Conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.do(x)

        # ResBlocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        # final layer
        x = self.global_avg_pool1(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x




