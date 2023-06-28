import torch
import torchvision
import torch.optim as optim
import torch.cuda as cuda
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import matplotlib.pyplot  as plt
import numpy as np
from os.path import exists
from Dataset import FramesDataset
from convNet import convNet
from UNET import UNET
from torch.utils.tensorboard import SummaryWriter


def train(LR,CHANNELS):    
    BATCH = 2
    EPOCHS = 10
    #LR = 1e-4
    #CHANNELS = [32,64,128,256,512]
    RUNID = f'runs/LR={LR}_CHANNELS={CHANNELS}__Mixed_Precission16__L1__loss_Encoder_BATCHNORM__DROPOUT=0.2'
    writer = SummaryWriter(RUNID)
    device = "cuda" if cuda.is_available() else "cpu"
    dir_dataset = r'C:\Users\Mau\Desktop\proyectos\Proyecti_Final\dataset2'
    dataset = FramesDataset(dir = dir_dataset, transform=transforms.ToTensor())
    model = convNet(channels=CHANNELS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    d_scaler = torch.cuda.amp.GradScaler(init_scale = float(2**64),growth_factor=float(10),backoff_factor= float(0.9))
    
    trainset, testset = random_split(dataset,[0.9,0.1])
    trainset = DataLoader(trainset, batch_size=BATCH, shuffle=True, pin_memory=True,num_workers=1)
    testset = DataLoader(testset, batch_size=1, shuffle=True)
    total_iters = len(trainset)
    lossFunction = torch.nn.L1Loss()
    BCE = torch.nn.BCEWithLogitsLoss()
    running_loss = 0.0
    iter = 0
    for epoch in range(EPOCHS):
        for data in trainset:
            (F1,F2,F3) = data
            with torch.cuda.amp.autocast(enabled=True):
                input = torch.cat([F1.to(device), F3.to(device)], dim=1)
                output = model(input)
                #pixelDif = BCE(torch.abs(output - F2.to(device)),torch.zeros_like(output))
                loss  = lossFunction(output.to(device), F2.to(device))
                #scalarLoss = loss.item()
                #loss = loss*100 + pixelDif
                scalarLoss = loss.item()
            model.zero_grad()
            d_scaler.scale(loss).backward()
            d_scaler.step(optimizer)
            d_scaler.update()
            iter += 1
            running_loss += scalarLoss 
            if (iter)%50==0:
                with torch.no_grad():
                    output = model(input)
                obj_grid = torchvision.utils.make_grid(F2)
                writer.add_image('objetivo', obj_grid,(iter+total_iters*epoch))
                res_grid = torchvision.utils.make_grid(output)
                writer.add_image('resultado', res_grid,(iter+total_iters*epoch))
                writer.add_scalar('loss',running_loss/50,(iter+total_iters*epoch))
                print(running_loss/50)
                running_loss=0
    writer.close()

if __name__ == '__main__':
    CHANNELS_LIST = [
        [64,128,256],
        [32,64,128,256],
        [32,64,128,256,512]
    ]
    LR_LIST = [10e-4]
    for channels in CHANNELS_LIST:
        for lr in LR_LIST:
            train(lr,channels)