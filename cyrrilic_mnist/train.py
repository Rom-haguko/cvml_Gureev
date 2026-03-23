import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import torch.optim as optim
from pathlib import Path
import cv2
import zipfile
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split

def scan_zip_archive(archive_path):
    with zipfile.ZipFile(archive_path, 'r') as arch:
        png_files = [item for item in arch.namelist() if item.endswith('.png') and '/' in item]
        unique_labels = sorted(list(set(item.split('/')[1] for item in png_files)))
        label_mapping = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    return png_files, label_mapping

class LettersZipDataset(Dataset):
    def __init__(self, path_to_zip, img_list, mapping, transforms_obj=None):
        self.transforms_obj = transforms_obj
        self.path_to_zip = path_to_zip
        self.img_list = img_list
        self.mapping = mapping
        self.zip_ref = zipfile.ZipFile(path_to_zip, 'r')

    def __len__(self):
        return len(self.img_list)
  
    def __getitem__(self, i):
        file_path = self.img_list[i]
        
        raw_bytes = np.frombuffer(self.zip_ref.read(file_path), np.uint8)
        img = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
        
        img = img[:, :, 3]
        img = np.expand_dims(img, axis=-1)
        
        cls_name = file_path.split('/')[1]
        lbl_idx = self.mapping[cls_name]
        
        if self.transforms_obj:
            img = self.transforms_obj(img)
        return img, lbl_idx

class CustomCyrillicNet(nn.Module):
    def __init__(self):
        super(CustomCyrillicNet, self).__init__()
        self.layer1_conv = nn.Conv2d(1, 32, 3, padding=1)
        self.layer1_bn = nn.BatchNorm2d(32)
        self.layer1_act = nn.ReLU()
        self.layer1_pool = nn.MaxPool2d(2, 2)
        
        self.layer2_conv = nn.Conv2d(32, 64, 3, padding=1)
        self.layer2_bn = nn.BatchNorm2d(64)
        self.layer2_act = nn.ReLU()
        self.layer2_pool = nn.MaxPool2d(2, 2)
        
        self.layer3_conv = nn.Conv2d(64, 128, 3, padding=1)
        self.layer3_bn = nn.BatchNorm2d(128)
        self.layer3_act = nn.ReLU()
        self.layer3_pool = nn.MaxPool2d(2, 2) 
        
        self.flat = nn.Flatten()
        self.dense1 = nn.Linear(128 * 3 * 3, 256) 
        self.dense_act = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.dense2 = nn.Linear(256, 34)

    def forward(self, val):
        val = self.layer1_pool(self.layer1_act(self.layer1_bn(self.layer1_conv(val))))
        val = self.layer2_pool(self.layer2_act(self.layer2_bn(self.layer2_conv(val))))
        val = self.layer3_pool(self.layer3_act(self.layer3_bn(self.layer3_conv(val))))
        
        val = self.flat(val)
        val = self.drop(self.dense_act(self.dense1(val)))
        val = self.dense2(val)
        return val

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    archive_file = base_dir / 'cyrillic.zip'
    weights_file = base_dir / "model.pth"
    
    
    computing_device = torch.device("cpu")

    paths_list, map_classes = scan_zip_archive(archive_file) 
    target_labels = [map_classes[f.split('/')[1]] for f in paths_list]

    train_files, test_files, train_targets, _ = train_test_split(
        paths_list, target_labels, test_size=0.2, random_state=42, stratify=target_labels
    )

    aug_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28, 28)),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)), 
        transforms.Normalize((0.5,), (0.5,))
    ])

    eval_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28, 28)),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data_train = LettersZipDataset(archive_file, train_files, map_classes, aug_transforms)
    data_test = LettersZipDataset(archive_file, test_files, map_classes, eval_transforms)

    loader_train = DataLoader(data_train, batch_size=64, shuffle=True)
    loader_test = DataLoader(data_test, batch_size=64, shuffle=False)

    nn_model = CustomCyrillicNet().to(computing_device)
    loss_function = nn.CrossEntropyLoss()
    opt = optim.Adam(nn_model.parameters(), lr=0.001)
    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10)

    loss_history = []
    acc_history = []

    for epoch_idx in range(15):
        nn_model.train()
        current_loss, correct_answers, total_items = 0.0, 0, 0

        for batch_imgs, batch_lbls in loader_train:
            batch_imgs, batch_lbls = batch_imgs.to(computing_device), batch_lbls.to(computing_device)
            
            opt.zero_grad()
            model_out = nn_model(batch_imgs)
            batch_loss = loss_function(model_out, batch_lbls)
            batch_loss.backward()
            opt.step()
            
            current_loss += batch_loss.item()
            preds = model_out.argmax(dim=1)
            total_items += batch_lbls.size(0)
            correct_answers += (preds == batch_lbls).sum().item()
        
        lr_scheduler.step()
        ep_loss = current_loss / len(loader_train)
        ep_acc = 100 * (correct_answers / total_items)
        loss_history.append(ep_loss)
        acc_history.append(ep_acc)
        

        print(f"Epoch: {epoch_idx+1}, Loss: {ep_loss:.4f}, Acc: {ep_acc:.2f}%")

    torch.save(nn_model.state_dict(), weights_file)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(121); plt.title("Loss"); plt.plot(loss_history)
    plt.subplot(122); plt.title("Accuracy"); plt.plot(acc_history)
    plt.savefig(base_dir / "train.png")
    plt.show()
    

    nn_model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for batch_imgs, batch_lbls in loader_test:
            batch_imgs, batch_lbls = batch_imgs.to(computing_device), batch_lbls.to(computing_device)
            model_out = nn_model(batch_imgs)
            preds = model_out.argmax(dim=1)
            val_total += batch_lbls.size(0)
            val_correct += (preds == batch_lbls).sum().item()

    print(f"Accuracy: {100 * val_correct / val_total:.2f}%")