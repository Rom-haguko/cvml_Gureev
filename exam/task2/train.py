#accuracy 98.6
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import torch.optim as optim
import time
from sklearn.model_selection import train_test_split

save_path = Path(__file__).parent
data_path = save_path / "chinese"


class LeNet5(nn.Module):

    def __init__(self, num_classes=15):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 120, 5, padding=2)

        self.fc1 = nn.Linear(120 * 8 * 8, 84)
        self.fc2 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class ChineseDataset(Dataset):

    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    
    def __getitem__(self, idx):
        img_path = self.images[idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        code = int(img_path.stem.split("_")[-1])
        label = code - 1

        if self.transform:
            image = self.transform(image)

        return image, label


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomAffine(
        degrees=8,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05),
        fill=0
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


if __name__ == "__main__":
    data_dir = data_path / "data" / "data"
    images = sorted(data_dir.glob("*.jpg"))

    labels = [
        int(img_path.stem.split("_")[-1]) - 1
        for img_path in images
    ]

    train_images, test_images = train_test_split(
        images,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    train_data = ChineseDataset(train_images, transform=train_transform)
    test_data = ChineseDataset(test_images, transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    model = LeNet5(num_classes=15).to("mps")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    t = time.perf_counter()
    best_acc = 0

    for epoch in range(40):
        model.train()
        for data, target in train_loader:
            data = data.to("mps")
            target = target.to("mps")

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to("mps")
                target = target.to("mps")

                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()

        acc = 100.0 * correct / len(test_data)
        print(f"Epoch {epoch + 1}, {acc=}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path / "lenet5.pth")
            # print(f"Best model saved, {best_acc=}")

    print(f"Best accuracy {best_acc}")
    print(f"Elapsed time {time.perf_counter() - t}")