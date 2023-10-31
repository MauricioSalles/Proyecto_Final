import cv2
import numpy as np
import NN_resorces.Dataset as dt
import torch
import torchvision.transforms as transforms
from NN_resorces.refine_flow import FlowModule
from NN_resorces.FlowNet import FlowNet
dataset = dt.Dataset('dataset')
len = dataset.__len__()
flow = FlowModule()
totalError = 0
totalErrorCluster = 0
totalErrorSector = 0
totalErrorRefine = 0

def coords_grid(batch, ht, wd, device):
        coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
        coords = torch.stack(coords[::-1], dim=0).float()
        return coords[None].repeat(batch, 1, 1, 1)

def warping(coords, img):
    N, C, H, W = img.shape
    coords = coords.permute(0, 2, 3, 1)
    xgrid, ygrid = coords.split([1,1], dim=3)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    output = torch.nn.functional.grid_sample(img, grid, align_corners=True)
    return output

def calcFrame(frame1 , frame2,weight):
    flow = FlowNet().to('cuda')
    flow.load_state_dict(torch.load(weight))
    flow.eval()
    transform = transforms.ToTensor()
    F1=torch.unsqueeze(transform(frame1).to('cuda'), dim=0)
    F3=torch.unsqueeze(transform(frame2).to('cuda'), dim=0)
    N, C, H, W = F1.shape
    coords = coords_grid(N, H, W, device='cuda')
    with torch.no_grad():
        flow = flow(F1.to('cuda'), F3.to('cuda'),5,False)
    coords = coords + flow
    output = warping(coords,F1.to('cuda'))

    return output


for i in range(100):
    print(i)
    transform = transforms.ToTensor()
    f1, f2, f3 = dataset.__getitem__(i)
    F3=torch.unsqueeze(transform(f3).to('cuda'), dim=0)
    F1=torch.unsqueeze(transform(f1).to('cuda'), dim=0)
    FlowNetBC = calcFrame(f1,f3,'./weights/FlowNetBC.pth')
    FlowNetUFlow = calcFrame(f1,f3,'./weights/FlowNetUFlow.pth')
    FlowNetBCError = np.str_(torch.abs(torch.mean(FlowNetBC-F3)*255))
    FlowNetUFlowError = np.str_(torch.abs(torch.mean(FlowNetUFlow-F3)*255))
    error = np.str_(torch.abs(torch.mean(F1-F3)*255))
print("error FlowNetBC:  "+ FlowNetBCError)  
print("error FlowNetUFlow:  "+ FlowNetUFlowError)  
print("error :  "+ error)  