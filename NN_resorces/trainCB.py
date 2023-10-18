import torch
import torchvision
import torch.optim as optim
import torch.cuda as cuda
import torchvision.transforms as transforms
import CustomLossesBC as cLoss
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from Dataset import Dataset
from FlowNet import FlowNet, coords_grid
from torch.utils.tensorboard import SummaryWriter


def inputWarp(img,deltaX,deltaY):
    N, C, H, W = img.shape
    coords = torch.meshgrid(torch.arange(H, device=img.device)+deltaY, torch.arange(W, device=img.device)+deltaX)
    coords = torch.stack(coords[::-1], dim=0).float()
    coords = coords[None].repeat(N, 1, 1, 1)
    return warping(coords,img)
    

def warping(coords, img):
    N, C, H, W = img.shape
    coords = coords.permute(0, 2, 3, 1)
    xgrid, ygrid = coords.split([1,1], dim=3)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    output = torch.nn.functional.grid_sample(img, grid, align_corners=True)
    return output

def train(LR, EPOCHS,RUNID, model,currentStep):    
    BATCH = 2
    iter = 5
    running_loss = 0.0
    step = currentStep
    
    writer = SummaryWriter(RUNID)
    device = "cuda" if cuda.is_available() else "cpu"
    dir_dataset = r'C:\Users\Mau\Desktop\proyectos\refineCNN\dataset'
    dataset = Dataset(dir = dir_dataset, transform=transforms.ToTensor())
    optimizer = optim.Adam(model.parameters(), lr=LR)
    trainset, testset = random_split(dataset,[0.9,0.1])
    trainset = DataLoader(trainset, batch_size=BATCH, shuffle=True, pin_memory=True,num_workers=1)
    testset = DataLoader(testset, batch_size=1, shuffle=True) 
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(EPOCHS):
        for data in trainset:
            (F1,F2,F3) = data
            warps = []
            N, C, H, W = F1.shape
            with torch.cuda.amp.autocast():
                coords = coords_grid(N, H, W, device=device)
                outputs = model(F1.to(device), F3.to(device),iter)
                for output in outputs:
                    warpCoords = coords + output
                    warps.append(warping(warpCoords,F1.to(device)))
                loss = cLoss.unsup_loss(outputs, warps, F3.to(device))
            scalarLoss = loss.item()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #loss.backward()
            #optimizer.step()
            #optimizer.zero_grad()
            step += 1
            running_loss += scalarLoss 
            if (step)%50==0:
                N, C, H, W = F1.shape
                coords = coords_grid(N, H, W, device=device)
                with torch.no_grad():
                    flow = model(F1.to(device), F3.to(device),iter,False)
                coords = coords + flow*0.5
                output = warping(coords,F1.to(device))
                obj_grid = torchvision.utils.make_grid(F2)
                writer.add_image('objetivo', obj_grid,(step))
                res_grid = torchvision.utils.make_grid(output)
                writer.add_image('resultado', res_grid,(step))
                writer.add_scalar('loss',running_loss/50,(step))
                print(running_loss/50)
                running_loss=0
    writer.close()
    return step

if __name__ == '__main__':

    LR_LIST = [2e-4,2e-5]
    EPOCH = [5,5]
    step = 0
    model = FlowNet().to("cuda" if cuda.is_available() else "cpu")
    RUNID = f'runs/FlowNetBC'
    for idx,lr in enumerate(LR_LIST):
        step = train(lr,EPOCH[idx],RUNID,model,step)
    torch.save(model.state_dict(), './weights/FlowNetBC.pth')
        