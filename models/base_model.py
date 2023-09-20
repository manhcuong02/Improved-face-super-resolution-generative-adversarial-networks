import torch
from torch import nn


def make_layer(k, channels):
    return nn.Sequential(
        nn.Conv2d(k * channels, channels, 3, 1, 1),
        nn.PReLU()
    )

class DenseBlock(nn.Module):
    def __init__(self, channels = 12):
        super(DenseBlock, self).__init__()
        
        layer_list = []
        
        for i in range(1, 16):
            layer_list.append(make_layer(i, channels))
            
        self.layer_list = nn.ModuleList(layer_list)
        
        self.conv_out = nn.Conv2d(16*channels, 16*channels, 3, 1, 1)
        
    def forward(self, x):
        '''x: batch, channels, height, width'''
        input_list = [x]
        for i, layer in enumerate(self.layer_list):
            x = layer(torch.cat(input_list, dim = 1))
            input_list.append(x) 
            
        x = self.conv_out(torch.cat(input_list, dim = 1))
        return x
    
class SubPixelConv(nn.Module):
    def __init__(self, channels = 64, upscale_factor = 2):
        super(SubPixelConv, self).__init__()

        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
            nn.PReLU(),
        )
    
    def forward(self, x):
        x = self.upsample_block(x)
        
        return x

class SNConv2d(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1,
        bias = True,
        padding_mode: str = 'zeros',
        device = None,
        dtype = None
    ):
        super().__init__()
        
        self.snconv = nn.utils.spectral_norm(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
                padding_mode,
                device,
                dtype
            )
        )
    
    def forward(self, x):
        
        return self.snconv(x)

class SNConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.block1 = nn.Sequential(
                SNConv2d(in_channels, in_channels, (3, 3), (2, 2), (1, 1), bias=False),
                nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(0.2, True)
            ) 
        
        self.block2 = nn.Sequential(
                SNConv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, True)
            ) 

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        
        return x