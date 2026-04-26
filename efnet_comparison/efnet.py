from pathlib import Path
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
root = Path(__file__).resolve().parent

train_tf = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255]),
])

val_tf = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255]),
])

train_p = root / "dataset" / "dataset" / "train"
val_p = root / "dataset" / "dataset" / "val"

train_data = ImageFolder(train_p, transform=train_tf)
val_data = ImageFolder(val_p, transform=val_tf)

train_batch = DataLoader(train_data, batch_size=32, shuffle=True)
val_batch = DataLoader(val_data, batch_size=32, shuffle=False)

def get_net(v):
    if v == "b0":
        net = torchvision.models.efficientnet_b0(weights='DEFAULT')
    elif v == "b1":
        net = torchvision.models.efficientnet_b1(weights='DEFAULT')
    elif v == "b2":
        net = torchvision.models.efficientnet_b2(weights='DEFAULT')
        
    for p in net.parameters():
        p.requires_grad = False
    
    in_f = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_f, 3)
    )
    return net.to(device)

if __name__ == "__main__":
    nets = ["b0", "b1", "b2"]
    res = {}
    total = 20

    for v in nets:
        model = get_net(v)
        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
        step_fn = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
        
        for ep in range(total):
            model.train()
            for x, y in train_batch:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                opt.step()
            
            model.eval()
            all_p, all_y = [], []
            ok, num = 0, 0
            with torch.no_grad():
                for x, y in val_batch:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    p = out.argmax(1)
                    all_p.extend(p.cpu().numpy())
                    all_y.extend(y.cpu().numpy())
                    ok += (p == y).sum().item()
                    num += y.size(0)
            
            step_fn.step()
            acc = ok / num
            print(f"{v} ep {ep+1} acc: {acc:.4f}")
            
            if ep == total - 1:
                res[v] = acc
                cm = confusion_matrix(all_y, all_p)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=train_data.classes, yticklabels=train_data.classes)
                plt.title(f"EfficientNet {v}")
                plt.tight_layout()
                plt.savefig(root / f"cm_{v}.png")
                plt.close()

    for k, v in res.items():
        print(f"Final {k}: {v*100:.2f}%")