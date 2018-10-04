import torch
import torch.nn as nn

from layer import BasicBlock

class ResNet(nn.Module):
    def __init__(self, num_blocks_list, num_classes=100):
        super(ResNet, self).__init__()
        self.curt_in_channels = 32

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32) 
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

        self.conv2_x = self._add_layers(32, num_blocks_list[0])
        self.conv3_x = self._add_layers(64, num_blocks_list[1], start_stride=2)
        self.conv4_x = self._add_layers(128, num_blocks_list[2], start_stride=2)
        self.conv5_x = self._add_layers(256, num_blocks_list[3], start_stride=2)
        
        self.maxpool = nn.MaxPool2d(4, stride=1)
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.dropout(self.relu(self.conv1(x)))

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.maxpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _add_layers(self, out_channels, num_blocks, start_stride=1):
        # Configure downsampling for the residual
        downsample = False
        if start_stride != 1 or self.curt_in_channels != out_channels:
            downsample = True
        
        # The first block may need downsampling
        layers = []
        layers.append(BasicBlock(self.curt_in_channels, out_channels,
                                 start_stride=start_stride, downsample=downsample))
        self.curt_in_channels = out_channels

        # The remaining blocks are all s=1, p=1
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.curt_in_channels, out_channels))
        
        return nn.Sequential(*layers)

