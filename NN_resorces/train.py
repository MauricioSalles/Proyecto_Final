import torch
import torchvision
import torch.optim as optim
import torch.cuda as cuda
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from Dataset import FramesDataset
from convNet import convNet,SimpleConvNet
from UNET import UNET
from torch.utils.tensorboard import SummaryWriter


def train(LR, EPOCHS,RUNID, model,step):    
    BATCH = 3
    writer = SummaryWriter(RUNID)
    device = "cuda" if cuda.is_available() else "cpu"
    dir_dataset = r'C:\Users\Mau\Desktop\proyectos\Proyecti_Final\dataset2'
    dataset = FramesDataset(dir = dir_dataset, transform=transforms.ToTensor())
    optimizer = optim.Adam(model.parameters(), lr=LR)
    trainset, testset = random_split(dataset,[0.9,0.1])
    trainset = DataLoader(trainset, batch_size=BATCH, shuffle=True, pin_memory=True,num_workers=1)
    testset = DataLoader(testset, batch_size=1, shuffle=True)
    lossFunction = torch.nn.L1Loss()
    running_loss = 0.0
    iter = step
    for epoch in range(EPOCHS):
        for data in trainset:
            (F1,F2,F3) = data
            output = model(F1.to(device), F3.to(device))
            pixelDif = torch.abs(output - F2.to(device))
            loss  = lossFunction(torch.abs(output - F2.to(device)),torch.zeros_like(output))
            scalarLoss = loss.item()
            model.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            iter += 1
            running_loss += scalarLoss 
            if (iter)%50==0:
                with torch.no_grad():
                    output = model(F1.to(device), F3.to(device))
                obj_grid = torchvision.utils.make_grid(F2)
                writer.add_image('objetivo', obj_grid,(iter))
                res_grid = torchvision.utils.make_grid(output)
                writer.add_image('resultado', res_grid,(iter))
                writer.add_scalar('loss',running_loss/50,(iter))
                print(running_loss/50)
                running_loss=0
    writer.close()
    return step

if __name__ == '__main__':
    CHANNELS_LIST = [
        [32,64,128,256]
    ]
    LR_LIST = [2e-5]
    EPOCH = [15,15]
    step = 0
    for channels in CHANNELS_LIST:
        model = SimpleConvNet(channels=channels).to("cuda" if cuda.is_available() else "cpu")
        RUNID = f'runs/SimpleConvNet-CHANNELS={channels}'
        for idx,lr in enumerate(LR_LIST):
            step = train(lr,EPOCH[idx],RUNID,model,step)
        torch.save(model.state_dict(), './weights/SimpleConvNet2e-5.pth')
        