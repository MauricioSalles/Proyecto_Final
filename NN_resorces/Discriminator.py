import torch.nn as nn
from torch import cat
import torch.utils.checkpoint as checkpoint
try:
    from .convBlock import ConvBlock                  
except ImportError:
    try:
        from convBlock import ConvBlock
    except ImportError:
        print("no existe modulo")

class Discriminator(nn.Module):
    """The quadratic model"""
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, padding_mode="reflect"),
            nn.Conv2d(64, 64, kernel_size=4, padding_mode="reflect"),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=4, padding_mode="reflect"),
            nn.Conv2d(256, 256, kernel_size=4, padding_mode="reflect"),
            nn.Conv2d(256, 512, kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=4, padding_mode="reflect"),
        )

    def forward(self, x, y): 
        x = cat([x, y], dim=1)
        return self.conv(x)