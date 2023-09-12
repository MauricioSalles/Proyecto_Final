import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
from Dataset import Dataset
from UNET import UNET
from Discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.utils import save_image

torch.backends.cudnn.benchmark = True

BATCH = 3
NUM_EPOCHS = 5
LR = 2e-5
device = "cuda" if cuda.is_available() else "cpu"
dir_dataset = '/content/drive/MyDrive/Proyecto_Final/Proyecto_Final/dataset2'

def main():
    disc = Discriminator(in_channels=9).to("cuda")
    gen = UNET(in_channels=6,channels=[64, 128, 256, 512]).to("cuda")
    opt_disc = optim.Adam(disc.parameters(), lr=LR)
    opt_gen = optim.Adam(gen.parameters(), lr=LR)
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.MSELoss(reduction='sum')

    trainset = Dataset(dir = dir_dataset, transform=transforms.ToTensor())
    train_loader = DataLoader(trainset, batch_size=BATCH, shuffle=True, num_workers=2)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        for data in train_loader:
            (F1,F2,F3) = data 
            F1 = F1.to(device)
            F2 = F2.to(device)
            F3 = F3.to(device)
            # Train Discriminator
            with torch.cuda.amp.autocast():
                input = torch.cat([F1, F3], dim=1)
                y_fake = gen(input)
                D_real = disc(input, F2)
                D_real_loss = BCE(D_real, torch.ones_like(D_real))
                D_fake = disc(input, y_fake.detach())
                D_fake_loss = BCE(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2
            disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()
            # Train generator
            with torch.cuda.amp.autocast():
                D_fake = disc(input.to(device), y_fake)
                G_fake_loss = BCE(D_fake, torch.ones_like(D_fake))
                L1 = L1_LOSS(y_fake, F2)
                G_loss = G_fake_loss + L1*100
            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()
        torch.save(disc.state_dict(), "weights/disc-conv.pth")
        torch.save(gen.state_dict(), "weights/gen-conv.pth")
        print(epoch)

if __name__ == "__main__":
    main()
