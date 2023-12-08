import torch
import torch.nn as nn
try:
    from .convBlock import ConvBlock   
    import NN_resorces.util as util             
except ImportError:
    try:
        import util
        from convBlock import ConvBlock
    except ImportError:
        print("no existe modulo")

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, channels, norm_fn='batch', stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        if norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(channels)
            self.norm2 = nn.BatchNorm2d(channels)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(channels)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(channels)
            self.norm2 = nn.InstanceNorm2d(channels)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(channels)

        if stride == 1:
            self.downsample = None
        else:    
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x+y)
    
class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
            
        if self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
            
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)
        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout2d(p=dropout)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        if self.training and self.dropout is not None:
            x = self.dropout(x)
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        return x
                
class FeatureExtractor(nn.Module):
    def __init__(self,in_channels,norm_fn):
        super(FeatureExtractor,self).__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels,   32, batch_norm=norm_fn, activation='relu'),
            nn.Conv2d(32,   32, kernel_size=2, stride=2, bias=False),
            ConvBlock(32,   64, batch_norm=norm_fn, activation='relu'),
            nn.Conv2d(64,   64, kernel_size=2, stride=2, bias=False),
            ConvBlock(64,   128, batch_norm=norm_fn, activation='relu'),
            nn.Conv2d(128,   256, kernel_size=2, stride=2, bias=False),
            ConvBlock(256,   256, activation='relu'),
        ) 

    def forward(self, x):
        x = self.encoder(x)
        x1, x2 = torch.split(x,[128,128],dim=1)
        return x1,x2  
     
class UpsampleFlow(nn.Module):
    def __init__(self):
        super(UpsampleFlow,self).__init__()
        self.decoder = nn.Sequential(
            ConvBlock(2,   2, batch_norm=True, activation='relu'),
            nn.ConvTranspose2d(2,   2, kernel_size=2, stride=2, bias=False),
            ConvBlock(2,   2, batch_norm=True, activation='relu'),
            nn.ConvTranspose2d(2,   2, kernel_size=2, stride=2, bias=False),
            ConvBlock(2,   2, batch_norm=True, activation='relu'),
            nn.ConvTranspose2d(2,   2, kernel_size=2, stride=2, bias=False),
        ) 

    def forward(self, x):
        x = self.decoder(x)
        return x 
     
def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

class FlowNet(nn.Module):
    def __init__(self, dropout=None):
        super(FlowNet,self).__init__()
        self.featuerExtract = BasicEncoder(output_dim=256, norm_fn='instance', dropout=dropout) 
        self.context = BasicEncoder(output_dim=256, norm_fn='batch', dropout=dropout)
        self.encoder = util.MotionEncoder()
        self.gru = util.ConvGRU(hidden_dim=128, input_dim=256)
        self.flow_head = util.FlowHead(128, hidden_dim=256)
        self.upsampleFlow = UpsampleFlow()

    def forward(self, x1,x2,iter,train=True):
        N, C, H, W = x1.shape
        flows = []
        x1 = 2 * (x1 / 255.0) - 1.0
        x2 = 2 * (x2 / 255.0) - 1.0
        if W%32 != 0:
            x1=nn.functional.pad(x1, (W%32,0), mode='constant', value=0)
            x2=nn.functional.pad(x2, (W%32,0), mode='constant', value=0)
        if H%32 != 0:
            x1=nn.functional.pad(x1, (0,0,H%32,0), mode='constant', value=0)
            x2=nn.functional.pad(x2, (0,0,H%32,0), mode='constant', value=0)
        hOrigin = H
        wOrigin = W
        N, C, H, W = x1.shape
        coords0 = coords_grid(N, H//8, W//8, device=x1.device)
        coords1 = coords_grid(N, H//8, W//8, device=x1.device)
        
        x1feat,x2feat = self.featuerExtract([x1, x2])
        x1context = self.context(x1)
        net, x1context = torch.split(x1context, [128, 128], dim=1)
        net = torch.tanh(net)
        x1context = torch.relu(x1context)
        
        correlation = util.CorrBlock(x1feat, x2feat)
        
        for itr in range(iter):
            flow = coords1 - coords0
            corr = correlation(coords1)
            motion_features = self.encoder(flow, corr)
            inp = torch.cat([x1context, motion_features], dim=1)
            net = self.gru(net, inp)
            delta_flow = self.flow_head(net)
            coords1 = coords1 + delta_flow
            if train:
                flo = self.upsampleFlow(coords1 - coords0)[:,:,:hOrigin,:wOrigin]
                flows.append(flo)
            
        if train:
            return flows
        else:
            return self.upsampleFlow(coords1 - coords0)[:,:,:hOrigin,:wOrigin]
