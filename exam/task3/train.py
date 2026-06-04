import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from skimage import draw
import torch.optim as optim
import time

save_path = Path(__file__).parent


class BoxDataset(Dataset):

    def __init__(self, count=12000, transform=None):
        self.count = count
        self.transform = transform

    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        image = np.zeros((256, 256), dtype="uint8")

        label = idx % 3

        if label == 0:
            image = self.draw_triangle(image)
        elif label == 1:
            image = self.draw_square(image)
        else:
            image = self.draw_circle(image)

        image = self.thick_line(image)

        if self.transform:
            image = self.transform(image)

        mask = image[0] > 0

        ys, xs = torch.where(mask)

        x1 = xs.min()
        y1 = ys.min()
        x2 = xs.max()
        y2 = ys.max()

        h, w = image.shape[1], image.shape[2]

        bbox = torch.tensor([
            x1 / w,
            y1 / h,
            x2 / w,
            y2 / h
        ], dtype=torch.float32)

        return image, bbox
        
    def thick_line(self, image):
        kernel = np.ones((5, 5), dtype="uint8")
        image = cv2.dilate(image, kernel, iterations=2)
        return image

    def draw_circle(self, image):
        radius = np.random.randint(35, 80)

        y = np.random.randint(radius + 20, 256 - radius - 20)
        x = np.random.randint(radius + 20, 256 - radius - 20)

        rr, cc = draw.circle_perimeter(y, x, radius, shape=image.shape)
        image[rr, cc] = 255

        return image

    def draw_square(self, image):
        size = np.random.randint(70, 150)

        y1 = np.random.randint(20, 256 - size - 20)
        x1 = np.random.randint(20, 256 - size - 20)

        y2 = y1 + size
        x2 = x1 + size

        rr, cc = draw.rectangle_perimeter(
            start=(y1, x1),
            end=(y2, x2),
            shape=image.shape
        )
        image[rr, cc] = 255

        return image

    def draw_triangle(self, image):
        size = np.random.randint(70, 150)

        y1 = np.random.randint(20, 256 - size - 20)
        x1 = np.random.randint(20, 256 - size - 20)

        top_x = x1 + np.random.randint(size // 3, 2 * size // 3)

        r = np.array([y1, y1 + size, y1 + size])
        c = np.array([top_x, x1, x1 + size])

        rr, cc = draw.polygon_perimeter(r, c, shape=image.shape)
        image[rr, cc] = 255

        return image


class BoxDetector(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*2*2, 512),
            nn.ReLU(),
        )

        self.bbox_head = nn.Linear(512, 4)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = self.bbox_head(x)
        x = torch.sigmoid(x)
        return x


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.RandomRotation(25),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.10, 0.10),
        scale=(0.8, 1.2),
        shear=10
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


if __name__ == "__main__":
    train_data = BoxDataset(count=12000, transform=train_transform)
    test_data = BoxDataset(count=3000, transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    model = BoxDetector().to("mps")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.SmoothL1Loss()

    t = time.perf_counter()
    best_loss = 1000

    for epoch in range(10):
        model.train()
        train_loss = 0

        for data, bbox in train_loader:
            data = data.to("mps")
            bbox = bbox.to("mps")

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, bbox)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        test_loss = 0

        with torch.no_grad():
            for data, bbox in test_loader:
                data = data.to("mps")
                bbox = bbox.to("mps")

                output = model(data)
                loss = criterion(output, bbox)
                test_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        test_loss = test_loss / len(test_loader)

        print(f"Epoch {epoch + 1}, train_loss={train_loss:.5f}, test_loss={test_loss:.5f}")

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), save_path / "bbox_model.pth")
            # print(f"Best model saved, {best_loss=}")

    print(f"Best loss {best_loss}")
    print(f"Elapsed time {time.perf_counter() - t}")