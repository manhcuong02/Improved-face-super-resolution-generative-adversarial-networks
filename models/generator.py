import math

import torch
from torch import nn

from .base_model import DenseBlock, SubPixelConv


class Generator(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 3, channels = 64, upscale = 4):
        super(Generator, self).__init__()
        
        if not math.log2(upscale) == int(math.log2(upscale)):
            raise ValueError(f"Upscale factor `{upscale}` is not support.")
        
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 12, 9, 1, 4),
                nn.PReLU()
            )

        self.dense_block = DenseBlock(channels = 12)
    
        self.bottle_neck = nn.Conv2d(12 * 16, channels, kernel_size = (1, 1), stride = 1)
        
        self.subpixel_conv = nn.Sequential(
            SubPixelConv(channels, upscale_factor = upscale // 2),
            SubPixelConv(channels, upscale_factor = upscale // 2)
        )
        
        self.out = nn.Conv2d(channels, out_channels, (9, 9), (1, 1), (4, 4))
        
    def forward(self, x) -> torch.Tensor:
        x = self.conv(x)
        
        x = self.dense_block(x)
        
        x = self.bottle_neck(x)
        
        x = self.subpixel_conv(x)
        
        x = self.out(x)
        
        return x

if __name__ == '__main__':
    model = Generator(3, 3, 64, 4)
    x = torch.rand(2, 3, 32, 32)
    y = model(x)
    print(y.shape)