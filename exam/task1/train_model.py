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


class LeNet5(nn.Module):

    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        #self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(16*5*5, 21)
        self.fc2 = nn.Linear(21, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        #x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class ShapesDataset(Dataset):

    def __init__(self, count=9000, transform=None):
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

        return image, label
        
    def thick_line(self, image):
        kernel = np.ones((5, 5), dtype="uint8")
        image = cv2.dilate(image, kernel, iterations=2) # морфолигическая операция, которая расширяет белые пиксели.
        # iterations=2 чтобы края были более гладкие
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


test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05), 
        scale=(0.9, 1.1), 
        shear=5                       
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


if __name__ == "__main__":
    train_data = ShapesDataset(count=9000, transform=train_transform)
    test_data = ShapesDataset(count=1500, transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    model = LeNet5().to("mps")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    t = time.perf_counter()
    best_acc = 0

    for epoch in range(5):
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

        if best_acc <= acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path / "lenet5.pth")
            # print(f"Best model saved, {best_acc=}")
    
    print(f"Elapsed time {time.perf_counter() - t}")