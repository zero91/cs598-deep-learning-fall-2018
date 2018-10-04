import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, start_stride=1, downsample=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=start_stride,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Project the x into correct dimension if needed
        self.downsample = downsample
        if self.downsample:
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=start_stride,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(out_channels)                
            )
    
    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.downsample:
            residual = self.projection(x)
        x += residual
        x = self.relu(x)

        return x
        


