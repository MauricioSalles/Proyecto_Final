import torch.nn as nn
from torch import cat
from torchvision.transforms.functional import rgb_to_grayscale
from torch.nn.functional import pad
try:
    from .convBlock import ConvBlock                  
except ImportError:
    try:
        from convBlock import ConvBlock
    except ImportError:
        print("no existe modulo")

class SimpleConvNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, channels=[15, 30, 60, 120]):
        super(SimpleConvNet, self).__init__()
        self.channels = channels
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.activation = nn.Sigmoid()
        for feature in channels:
            self.encoders.append(ConvBlock(in_channels, feature,batch_norm=False,dropout=0.2))
            in_channels = feature
            
        self.middle = ConvBlock(channels[-1], channels[-1]*2,dropout=0.2)

        for feature in reversed(channels):

            self.decoders.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.decoders.append(ConvBlock(feature, feature,dropout=0.0))
            
        self.finalConv=ConvBlock(channels[0], out_channels,activation='hardsigmoid')

    def forward(self, x1,x2):
        x = cat([x1,x2], dim=1)
        dimension = []
        for idx,encoder in enumerate(self.encoders):
            x = encoder(x)
            b,c,w,h = x.shape
            dimension.append((w,h))
            padY = h%2
            padX = w%2
            x = pad(x, (0,padY,0,padX), "constant",value=0)
            x = self.pool(x)
            
        x = self.middle(x)
        dimension = dimension[::-1]
        for idx in range(0,len(self.decoders),2):
                x = self.decoders[idx](x)
                w,h = dimension[idx//2]
                x = x[:,:,:w,:h]
                x = self.decoders[idx+1](x)
        return self.finalConv(x)
    
    
class convNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=3, channels=[15, 30, 60, 120]):
        super(convNet, self).__init__()
        self.channels = channels
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.activation = nn.Sigmoid()
        self.ColorConv = nn.Conv2d(3,16,kernel_size=1)
        for feature in channels:
            self.encoders.append(ConvBlock(in_channels, feature,batch_norm=False,dropout=0.2))
            in_channels = feature
            
        self.middle = ConvBlock(channels[-1], channels[-1]*2,dropout=0.2)

        for feature in reversed(channels):

            self.decoders.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.decoders.append(ConvBlock(feature+32, feature,dropout=0.0))
            
        self.finalConv=ConvBlock(channels[0], out_channels,activation='hardsigmoid')

    def forward(self, x1,x2):
        x1color = self.ColorConv(x1)
        x2color = self.ColorConv(x2)
        x1gray = rgb_to_grayscale(x1)
        x2gray = rgb_to_grayscale(x2)
        x = cat([x1gray,x2gray], dim=1)
        xColorList = []
        xColorList.append(cat([x1color,x2color], dim=1))
        dimension = []
        for idx,encoder in enumerate(self.encoders):
            x = encoder(x)
            b,c,w,h = x.shape
            dimension.append((w,h))
            padY = h%2
            padX = w%2
            x = pad(x, (0,padY,0,padX), "constant",value=0)
            x = self.pool(x)
            xColor = xColorList[idx]
            xColor = pad(xColor, (0,padY,0,padX), "constant",value=0)
            xColorList.append(self.pool(xColor))
            
        x = self.middle(x)
        dimension = dimension[::-1]
        xColorList = xColorList[::-1]
        for idx in range(0,len(self.decoders),2):
                x = self.decoders[idx](x)
                w,h = dimension[idx//2]
                x = x[:,:,:w,:h]
                xColorList[1+idx//2] = xColorList[1+idx//2][:,:,:w,:h]
                x = self.decoders[idx+1](cat([x,xColorList[1+idx//2]],dim=1))
        return self.finalConv(x)