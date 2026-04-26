import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from pathlib import Path
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

classes = ["square", "circle", "triangle"]


class ShapesDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.images = []

        for cls_name in classes:
            img_dir = root / cls_name / "images"
            label_dir = root / cls_name / "labels"

            for img_path in sorted(img_dir.glob("*.png")):
                label_path = label_dir / (img_path.stem + ".txt")
                self.images.append((img_path, label_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label_path = self.images[idx]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            tensor = self.transform(Image.fromarray(img))
        else:
            tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        cls, x, y, w, h = map(float, label_path.read_text().split())
        bbox = torch.tensor([x, y, w, h], dtype=torch.float32)

        return tensor, int(cls), bbox


class SimpleDetector(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
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

            nn.AdaptiveAvgPool2d(2)
        )

        self.cls_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.bbox_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )

        self.cls_head = nn.Linear(128, num_classes)
        self.bbox_head = nn.Linear(512, 4)

    def forward(self, x):
        features = self.backbone(x)

        cls_features = self.cls_fc(features)
        bbox_features = self.bbox_fc(features)

        cls_pred = self.cls_head(cls_features)
        bbox_pred = torch.sigmoid(self.bbox_head(bbox_features))

        return cls_pred, bbox_pred


def giou_loss(pred, target):
    p_x1 = pred[:, 0] - pred[:, 2] / 2
    p_y1 = pred[:, 1] - pred[:, 3] / 2
    p_x2 = pred[:, 0] + pred[:, 2] / 2
    p_y2 = pred[:, 1] + pred[:, 3] / 2

    t_x1 = target[:, 0] - target[:, 2] / 2
    t_y1 = target[:, 1] - target[:, 3] / 2
    t_x2 = target[:, 0] + target[:, 2] / 2
    t_y2 = target[:, 1] + target[:, 3] / 2

    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area_p = (p_x2 - p_x1).clamp(min=0) * (p_y2 - p_y1).clamp(min=0)
    area_t = (t_x2 - t_x1).clamp(min=0) * (t_y2 - t_y1).clamp(min=0)

    union = area_p + area_t - inter
    iou = inter / (union + 1e-7)

    c_x1 = torch.min(p_x1, t_x1)
    c_y1 = torch.min(p_y1, t_y1)
    c_x2 = torch.max(p_x2, t_x2)
    c_y2 = torch.max(p_y2, t_y2)

    area_c = (c_x2 - c_x1).clamp(min=0) * (c_y2 - c_y1).clamp(min=0)

    giou = iou - (area_c - union) / (area_c + 1e-7)

    return (1 - giou).mean()


def detection_loss(cls_pred, bbox_pred, cls_targets, bbox_targets, lambda_bbox=100.0):
    loss_cls = F.cross_entropy(cls_pred, cls_targets)
    loss_bbox = F.smooth_l1_loss(bbox_pred, bbox_targets)
    loss_bbox += giou_loss(bbox_pred, bbox_targets)

    return loss_cls + lambda_bbox * loss_bbox, loss_cls, loss_bbox


transform = transforms.Compose([
    transforms.ToTensor()
])

dataset_name = "shapes_dataset_random"
root = Path(__file__).parent / "shapes" / dataset_name

train_ds = ShapesDataset(root / "train", transform=transform)
val_ds = ShapesDataset(root / "val", transform=transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

model = SimpleDetector(num_classes=len(classes)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

epochs = 30
accuracy_threshold = 0.95
save_path = root / "best.pt"

history = defaultdict(list)
best_acc = 0.0

for epoch in range(1, epochs + 1):
    model.train()

    train_loss = 0.0
    train_cls = 0.0
    train_box = 0.0

    for images, cls_t, bbox_t in train_loader:
        images = images.to(device)
        cls_t = cls_t.to(device)
        bbox_t = bbox_t.to(device)

        optimizer.zero_grad()

        cls_pred, bbox_pred = model(images)
        loss, loss_cls, loss_bbox = detection_loss(cls_pred, bbox_pred, cls_t, bbox_t)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_cls += loss_cls.item()
        train_box += loss_bbox.item()

    scheduler.step()

    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, cls_t, bbox_t in val_loader:
            images = images.to(device)
            cls_t = cls_t.to(device)
            bbox_t = bbox_t.to(device)

            cls_pred, bbox_pred = model(images)
            loss, _, _ = detection_loss(cls_pred, bbox_pred, cls_t, bbox_t)

            val_loss += loss.item()
            correct += (cls_pred.argmax(1) == cls_t).sum().item()
            total += cls_t.size(0)

    train_loss /= len(train_loader)
    train_cls /= len(train_loader)
    train_box /= len(train_loader)
    val_loss /= len(val_loader)
    val_acc = correct / total

    history["train_loss"].append(train_loss)
    history["train_cls"].append(train_cls)
    history["train_box"].append(train_box)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), save_path)

    print(
        f"Epoch {epoch:3d}/{epochs} "
        f"train={train_loss:.4f} "
        f"val={val_loss:.4f} "
        f"acc={val_acc:.4f}"
    )

    if val_acc >= accuracy_threshold:
        break

print(f"Лучшее accuracy: {best_acc:.4f}")

plt.figure()
plt.subplot(121)
plt.title("Loss")
plt.plot(history["train_loss"], label="train")
plt.plot(history["val_loss"], label="val")

plt.subplot(122)
plt.title("Accuracy")
plt.plot(history["val_acc"], label="val acc", color="green")

plt.legend()
plt.tight_layout()
plt.show()

def show_predictions(loader, model, n=8):
    model.eval()

    images, cls_t, bbox_t = next(iter(loader))
    images = images.to(device)

    with torch.no_grad():
        cls_pred, bbox_pred = model(images)

    preds = cls_pred.argmax(1).cpu()
    bbox_pred = bbox_pred.cpu()

    fig, axes = plt.subplots(2, n // 2, figsize=(16, 8))

    for i, ax in enumerate(axes.flat):
        img_np = images[i].cpu().numpy().transpose(1, 2, 0)

        h, w = img_np.shape[:2]

        cx, cy, bw, bh = bbox_t[i].numpy()
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h

        ax.add_patch(
            Rectangle(
                (x1, y1),
                bw * w,
                bh * h,
                linewidth=2,
                edgecolor="blue",
                facecolor="none"
            )
        )

        cx, cy, bw, bh = bbox_pred[i].numpy()
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h

        ax.add_patch(
            Rectangle(
                (x1, y1),
                bw * w,
                bh * h,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                linestyle="--"
            )
        )

        real_name = classes[cls_t[i]]
        pred_name = classes[preds[i]]

        ax.set_title(f"Real: {real_name}  Predicted: {pred_name}", fontsize=9)
        ax.imshow(img_np)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


show_predictions(val_loader, model)