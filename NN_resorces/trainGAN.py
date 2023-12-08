import torch
import torchvision
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
import torchvision.transforms as transforms
from Dataset import Dataset
from UNET import UNET
from Discriminator import Discriminator
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.benchmark = True


device = "cuda" if cuda.is_available() else "cpu"
dir_dataset = '/content/drive/MyDrive/Proyecto_Final/Proyecto_Final/dataset2'

def train(LR, EPOCHS,RUNID,disc,gen,currentStep):    
    BATCH = 1
    iter = 5
    running_loss = 0.0
    step = currentStep
    
    writer = SummaryWriter(RUNID)
    device = "cuda" if cuda.is_available() else "cpu"
    dir_dataset = r'C:\Users\Mau\Desktop\proyectos\refineCNN\dataset'
    dataset = Dataset(dir = dir_dataset, transform=transforms.ToTensor())
    trainset, testset = random_split(dataset,[0.9,0.1])
    trainset = DataLoader(trainset, batch_size=BATCH, shuffle=True, pin_memory=True,num_workers=1)
    opt_disc = optim.Adam(disc.parameters(), lr=LR)
    opt_gen = optim.Adam(gen.parameters(), lr=LR)
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.MSELoss(reduction='sum')
    testset = DataLoader(testset, batch_size=1, shuffle=True) 
    for epoch in range(EPOCHS):
        for data in trainset:
            (F1,F2,F3) = data 
            F1 = F1.to(device)
            F2 = F2.to(device)
            F3 = F3.to(device)
            # Train Discriminator
            input = torch.cat([F1, F3], dim=1)
            y_fake = gen(input)
            D_real = disc(input, F2)
            D_real_loss = BCE(D_real, torch.ones_like(D_real))
            D_fake = disc(input, y_fake.detach())
            D_fake_loss = BCE(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2
            disc.zero_grad()
            D_loss.backward()
            opt_disc.step()
            opt_disc.zero_grad()
            # Train generator
            D_fake = disc(input.to(device), y_fake)
            G_fake_loss = BCE(D_fake, torch.ones_like(D_fake))
            L1 = L1_LOSS(y_fake, F2)
            G_loss = G_fake_loss + L1*100
            gen.zero_grad()
            G_loss.backward()
            opt_gen.step()
            opt_gen.zero_grad()
            scalarLoss = L1.item()
            step += 1
            running_loss += scalarLoss 
            if (step)%50==0:
                with torch.no_grad():
                    input = torch.cat([F1, F3], dim=1)
                    output = gen(input)
                obj_grid = torchvision.utils.make_grid(F2)
                writer.add_image('objetivo', obj_grid,(step))
                res_grid = torchvision.utils.make_grid(output)
                writer.add_image('resultado', res_grid,(step))
                writer.add_scalar('loss',running_loss/50,(step))
                print(running_loss/50)
                running_loss=0
    return step

if __name__ == '__main__':

    LR_LIST = [2e-4,2e-5]
    EPOCH = [10,10]
    step = 0
    disc = Discriminator(in_channels=9).to("cuda")
    gen = UNET(in_channels=6,channels=[64, 128, 256, 512]).to("cuda")
    RUNID = f'runs/GAN'
    for idx,lr in enumerate(LR_LIST):
        step = train(lr,EPOCH[idx],RUNID,disc,gen,step)
    torch.save(disc.state_dict(), './weights/disc.pth')
    torch.save(gen.state_dict(), './weights/gen.pth')
        