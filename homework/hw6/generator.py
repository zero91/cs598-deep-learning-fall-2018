import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.in_dim = 4
        self.out_channels = 196

        self.fc1 = nn.Sequential(
            nn.Linear(100, 196*4*4),
            nn.BatchNorm1d(196*4*4)
        )

        self.conv1 = self._add_transpose_block()
        self.conv2 = self._add_conv_block()
        self.conv3 = self._add_conv_block()
        self.conv4 = self._add_conv_block()
        self.conv5 = self._add_transpose_block()
        self.conv6 = self._add_conv_block()
        self.conv7 = self._add_transpose_block()

        self.conv8 = nn.Conv2d(self.out_channels, 3, 3, padding=1, stride=1)
        self.tanh = nn.Tanh()

    def forward(self, x):    
        x = self.fc1(x)
        # Reshape the 196*4*4 vector into (batch_size, 196, 4, 4)
        x = x.view(-1, 196, 4, 4)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.conv8(x)
        x = self.tanh(x)

        return x
 
    def _add_conv_block(self):
        """Create conv block for each layer except the last one"""
        # conv2d -> batch norm -> relu
        block = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, stride=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        return block
    
    def _add_transpose_block(self):
        # convtranspose2d -> batch norm -> relu
        block = nn.Sequential(
            nn.ConvTranspose2d(self.out_channels, self.out_channels, 4, padding=1, stride=2),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        return block