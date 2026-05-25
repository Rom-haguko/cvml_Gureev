import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import time

path = Path("roads")

class RoadsDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.images_path = path / "images"
        self.masks_path = path / "masks"
        self.images = sorted(list(self.images_path.glob("*.png")))
        self.masks = sorted(list(self.masks_path.glob("*.png")))
        self.len = len(self.images)
        
        self.resize = transforms.Resize((288, 512))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")
        
        image = self.resize(image)
        mask = self.resize(mask)

        image = np.array(image) / 255.
        mask = np.array(mask, dtype="f4")
        
        mask = (mask == 82).astype("f4")
        mask = np.expand_dims(mask, axis=0)
        
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=2)
            
        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() # Формат: C, H, W
        mask = torch.from_numpy(mask.copy())
        return image, mask

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downscale = nn.ModuleList()
        self.upscale = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        for n in features:
            self.downscale.append(DoubleConv(in_channels, n))
            in_channels = n

        for n in reversed(features):
            self.upscale.append(nn.ConvTranspose2d(n * 2, n, 2, 2))
            self.upscale.append(DoubleConv(n*2, n))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.result = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for ds in self.downscale:
            x = ds(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skips = skips[::-1]
        for idx in range(0, len(self.upscale), 2):
            x = self.upscale[idx](x)
            skip = skips[idx // 2]
            cx = torch.cat((skip, x), dim=1)
            x = self.upscale[idx+1](cx)
            
        return self.result(x)

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_sig = torch.sigmoid(pred)
        p_area = pred_sig.view(-1)
        t_area = target.view(-1)
        intersection = (p_area * t_area).sum()
        return 1 - (2 * intersection + 1) / (p_area.sum() + t_area.sum() + 1)

if __name__ == '__main__':
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Обучение на устройстве: {device}")

    ds = RoadsDataset(path)
    train_dataloader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

    model = UNet(in_channels=3, out_channels=1).to(device)

    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        epoch_train_loss = 0.0
        
        for imgs, masks in train_dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            output = model(imgs)
            
            loss = criterion(output, masks) 
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        print(f"Эпоха [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}")

    torch.save(model.state_dict(), "unet_road.pth")