import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,batch_norm=False, activation='none',dropout=0.0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
        )
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        match activation:
            case 'relu':
                self.activation = nn.LeakyReLU(0.2)
            case 'sigmoid':
                self.activation = nn.Sigmoid()
            case 'hardsigmoid':
                self.activation = nn.Hardsigmoid()
            case 'none':
                self.activation = lambda x:x
        self.batch_norm = batch_norm
        self.normalize = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x =  self.activation(x)
        if self.batch_norm:
            x = self.normalize(x)
        return x