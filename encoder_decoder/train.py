import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch import nn
import random
import string
import time
import torch.optim as optim


class ImageDataset(Dataset):

    def __init__(self, n=200, size=128, mode=1):
        super().__init__()
        self.n = n
        self.size = size
        self.mode= mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5],std=[0.5])
        ])

    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        image = Image.new('L', (self.size, self.size),
                            color=255
                            )
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        match self.mode:
            case 1:
                text = "ABC"
                x = random.randint(10, self.size - 50)
                y = random.randint(10, self.size - 50)
            case 2:
                text = self._get_randtext(3)
                x = 30
                y = 30
            case 3:
                text = self._get_randtext(random.randint(1, 5))
                x = 30
                y = 30
            case 4:
                text = self._get_randtext(random.randint(1, 5))
                x = random.randint(10, self.size - 50)
                y = random.randint(10, self.size - 50)


        draw.text((x, y), text, fill=0, font=font)
        tensor = self.transform(image)
        return tensor, tensor
    
    def _get_randtext(self,length):
        return ''.join(random.choices(string.ascii_uppercase, k=length))
        
# ds = ImageDataset(2000, 256)
# print(ds[0][0].shape)
# plt.imshow(ds[0][0][0])
# plt.show()
        


class Encoder(nn.Module):
    def __init__(self,latent = 512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,kernel_size= 4,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32,64,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.bottleneck = nn.Linear(256 * 16 * 16,latent)

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.bottleneck(x)
        return x
    

class Decoder(nn.Module):
    def __init__(self,latent = 512):
        super().__init__()
        self.bottleneck = nn.Linear(latent,256 * 16 * 16)  

        self.features = nn.Sequential(
            nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32,1,kernel_size=4,stride=2,padding=1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.bottleneck(x)
        x = x.view(x.size(0),256,16,16)
        x = self.features(x)
        return x
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

encoder = Encoder()
decoder = Decoder()


if __name__ == '__main__':
    device = torch.device("mps")
    total_epochs = 10
    
    for curr_mode in range(1, 5):
        print(f"start: {curr_mode}")

        encoder = Encoder().to(device)
        decoder = Decoder().to(device)
        
        dataset = ImageDataset(n=20000, size=256, mode=curr_mode)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=10)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))

        encoder.train()
        decoder.train()

        for epoch in range(total_epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            
            for imgs, _ in dataloader:
                imgs = imgs.to(device)
                
                optimizer.zero_grad()
                latent = encoder(imgs)
                output = decoder(latent)
                
                loss = criterion(output, imgs) 
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{total_epochs}\nLoss: {avg_loss:.4f}\nTime: {time.time()-epoch_start:.2f}s\n")

        torch.save(encoder.state_dict(), f"encoder_{curr_mode}.pth")
        torch.save(decoder.state_dict(), f"decoder_{curr_mode}.pth")



