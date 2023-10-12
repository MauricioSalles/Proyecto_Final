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
        
class FeatureExtractor(nn.Module):
    def __init__(self,in_channels):
        super(FeatureExtractor,self).__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels,   32, batch_norm=True, activation='relu'),
            nn.Conv2d(32,   32, kernel_size=2, stride=2, bias=False),
            ConvBlock(32,   64, batch_norm=True, activation='relu'),
            nn.Conv2d(64,   64, kernel_size=2, stride=2, bias=False),
            ConvBlock(64,   128, batch_norm=True, activation='relu'),
            nn.Conv2d(128,   128, kernel_size=2, stride=2, bias=False),
            ConvBlock(128,   256, batch_norm=True, activation='relu'),
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
    def __init__(self):
        super(FlowNet,self).__init__()
        self.featuerExtract = FeatureExtractor(6)
        self.context = FeatureExtractor(3)
        self.encoder = util.MotionEncoder()
        self.gru = util.ConvGRU(hidden_dim=128, input_dim=256)
        self.flow_head = util.FlowHead(128, hidden_dim=256)
        self.upsampleFlow = UpsampleFlow()

    def forward(self, x1,x2,iter,train=True):
        N, C, H, W = x1.shape
        flows = []
        x1 = 2 * (x1 / 255.0) - 1.0
        x2 = 2 * (x2 / 255.0) - 1.0
        if W%16 != 0:
            x1=nn.functional.pad(x1, (W%16,0), mode='constant', value=0)
            x2=nn.functional.pad(x2, (W%16,0), mode='constant', value=0)
        if H%16 != 0:
            x1=nn.functional.pad(x1, (0,0,H%16,0), mode='constant', value=0)
            x2=nn.functional.pad(x2, (0,0,H%16,0), mode='constant', value=0)
        hOrigin = H
        wOrigin = W
        N, C, H, W = x1.shape
        coords0 = coords_grid(N, H//8, W//8, device=x1.device)
        coords1 = coords_grid(N, H//8, W//8, device=x1.device)
        
        x1feat,x2feat = self.featuerExtract(torch.cat([x1, x2], dim=1))
        net, x1context = self.context(x1)
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
