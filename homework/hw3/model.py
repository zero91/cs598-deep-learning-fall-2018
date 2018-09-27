import torch
import torch.nn as nn

class DeepCNN(nn.Module):
    def __init__(self):
        """Deep CNN model based on VGG16
        See the original paper: https://arxiv.org/abs/1409.1556
        """

        super(DeepCNN, self).__init__()
        self.cnov = self._add_conv_layers()
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        """Forward step which will be called directly
        by PyTorch
        """

        x = self.cnov(x)

        # Reshape tensor to match the fc dimensions.
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _add_conv_layers(self):
        # Network structures for CONV block and POOL block.
        out_channels_list = [
            64, 64, 'pool',
            128, 128, 'pool',
            256, 256, 256, 'pool',
            512, 512, 512, 'pool',
            512, 512, 512, 'pool'
        ]

        layers = []
        in_channels = 3

        # Build.
        for out_channels in out_channels_list:
            if out_channels == 'pool':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=3,
                    stride=1,
                    padding=1))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels

        return nn.Sequential(*layers)