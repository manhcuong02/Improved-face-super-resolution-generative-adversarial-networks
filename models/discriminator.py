import torch
from torch import nn

from .base_model import SNConvBlock


class Discriminator(nn.Module):
    def __init__(self, in_channels= 3, channels = 64):
        super().__init__()
        
        # batch, in_channels, 128, 128
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, channels, 3, 2, 1, bias = False),
                nn.LeakyReLU(0.2, True)
            )
        
        # batch, channels, 64, 64
        self.block1 = SNConvBlock(channels, 2*channels)
        
        # batch, 2*channels, 32, 32
        self.block2 = SNConvBlock(2*channels, 4*channels)
        
        # batch, 4*channels, 16, 16
        self.block3 = SNConvBlock(4*channels, 8*channels)
        
        # batch, 8*channels, 8, 8
        self.fully_connected = nn.Sequential(
            nn.Linear(8*channels*8*8, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 2),
            nn.Sigmoid()
        )
        
    def forward(self, x) -> torch.Tensor:
        '''x: batch, channels, height, width'''

        batch_size = x.shape[0]

        x = self.conv1(x)
        
        x = self.block1(x)
        
        x = self.block2(x)
        
        x = self.block3(x)
        
        x = x.view(batch_size, -1)
        
        x = self.fully_connected(x)
        
        return x
    