import torch.nn as nn
from torch.nn.functional import pad
from torch import no_grad,cuda,cat, load
from os.path import exists
try:
    from .convBlock import ConvBlock                  
except ImportError:
    try:
        from convBlock import ConvBlock
    except ImportError:
        print("no existe modulo")

class FeatureExtraction(nn.Module):
    def __init__(self, in_channels,channels=[32,64,128,256]):
        super(FeatureExtraction, self).__init__()
        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.first = ConvBlock(in_channels, channels[0])
        for channel in channels:
            self.blocks.append(ConvBlock(channel, channel))
        for channel in channels:
            self.downs.append(nn.Conv2d(channel, channel*2, kernel_size=2, stride=2))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        output = []
        x = self.first(x)
        for id in range(len(self.blocks)):
            x = self.blocks[id](x)
            output.append(x)
            x = self.downs[id](x)
        return output
    
class ProcessImage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ProcessImage, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.up = nn.ConvTranspose2d(out_channels, out_channels//2, kernel_size=2, stride=2)
    def forward(self, x):
        x = self.conv(x)
        up = self.up(x)
        return x, up.detach()
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.model = FeatureExtraction(3).to(self.device)
        self.level1 = ProcessImage(96,3).to(self.device)
        self.level2 = ProcessImage(192,64).to(self.device)
        self.level3 = ProcessImage(384,128).to(self.device)
        self.level4 = ProcessImage(512,256).to(self.device)
        if exists("./weights/FeatureExtraction.pth"):
            self.model.load_state_dict(load('./weights/FeatureExtraction.pth'))
            self.level1.load_state_dict(load('./weights/level1.pth'))
            self.level2.load_state_dict(load('./weights/level2.pth'))
            self.level3.load_state_dict(load('./weights/level3.pth'))
            self.level4.load_state_dict(load('./weights/level4.pth'))
        
    def forward(self,F1,F3):
        b,c,h,w = F1.shape
        if h%8>0 or w%8>0:
            F1 = pad(F1, (0,0,h%8,w%8), "constant", 0)
            F3 = pad(F3, (0,0,h%8,w%8), "constant", 0)
        with no_grad():
            features1 = self.model(F1.to(self.device))
            features3 = self.model(F3.to(self.device))
        input1= cat([features1[0], features3[0]], dim=1)
        input2= cat([features1[1], features3[1]], dim=1)
        input3= cat([features1[2], features3[2]], dim=1)
        input4= cat([features1[3], features3[3]], dim=1)
        _, up4 = self.level4(input4)
        _, up3 = self.level3(cat([input3, up4], dim=1))
        _, up2 = self.level2(cat([input2, up3], dim=1))
        out1, _ = self.level1(cat([input1, up2], dim=1))
        return out1