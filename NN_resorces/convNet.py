import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch.nn.functional import pad
try:
    from .convBlock import ConvBlock                  
except ImportError:
    try:
        from convBlock import ConvBlock
    except ImportError:
        print("no existe modulo")

class convNet(nn.Module):
    """The quadratic model"""
    def __init__(self, in_channels=6, out_channels=3, channels=[15, 30, 60, 120]):
        super(convNet, self).__init__()
        self.channels = channels
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activation = nn.Sigmoid()
        self.featureExtraction = nn.ModuleList()
        self.featureExtraction.requires_grad = False
        
        for feature in channels:
            self.encoders.append(ConvBlock(in_channels, feature))
            in_channels = feature
            

        for feature in reversed(channels):
            self.decoders.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.decoders.append(ConvBlock(feature, feature))
            
        self.decoders.append(ConvBlock(channels[0], out_channels))

        self.middle = ConvBlock(channels[-1], channels[-1]*2)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        b,c,w,h = x.shape
        padY = h%2**(len(self.channels)+1)
        padX = w%2**(len(self.channels)+1)
        x = pad(x, (0,padY,0,padX), "constant", 1)
        for encoder in self.encoders:
            x = encoder(x)
            x = self.pool(x)

        x = self.middle(x)
        
        for decoder in self.decoders:
            x = decoder(x)
        x = self.activation(x)
        x = x[:,:,:w,:h]
        return x