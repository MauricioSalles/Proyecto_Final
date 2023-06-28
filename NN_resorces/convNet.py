import torch.nn as nn
from torch.utils.checkpoint import checkpoint
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
        
        for feature in channels:
            self.encoders.append(ConvBlock(in_channels, feature,batch_norm=True,dropout=0.2))
            in_channels = feature
            
        self.middle = ConvBlock(channels[-1], channels[-1]*2,dropout=0.2)

        for feature in reversed(channels):

            self.decoders.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.decoders.append(ConvBlock(feature, feature,dropout=0.2))
            
        self.decoders.append(ConvBlock(channels[0], out_channels,activation='sigmoid'))

    def forward(self, x):
        dimension = []
        for encoder in self.encoders:
            x = encoder(x)
            b,c,w,h = x.shape
            dimension.append((w,h))
            padY = h%2
            padX = w%2
            
            x = pad(x, (0,padY,0,padX), "constant",value=0)
            x = self.pool(x)
        x = self.middle(x)
        dimension = dimension[::-1]
        for idx, decoder in enumerate(self.decoders):
                x = decoder(x)
                if idx%2 == 0 and idx//2<(len(dimension)):
                    w,h = dimension[idx//2]
                    x = x[:,:,:w,:h]
        return x