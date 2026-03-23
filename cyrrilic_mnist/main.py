import torch, zipfile
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from train import CustomCyrillicNet, LettersZipDataset, scan_zip_archive

base_dir = Path(__file__).parent


computing_device = torch.device("cpu") 

archive_file = base_dir / 'cyrillic.zip'
weights_file = base_dir / "model.pth"

paths_list, map_classes = scan_zip_archive(archive_file)
idx_to_str = {val: key for key, val in map_classes.items()}
target_labels = [map_classes[p.split('/')[1]] for p in paths_list]

_, eval_paths = train_test_split(paths_list, test_size=0.2, random_state=42, stratify=target_labels)

eval_dataloader = DataLoader(
    LettersZipDataset(archive_file, eval_paths, map_classes, transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28, 28)),
        transforms.Normalize((0.5,), (0.5,))
    ])), 
    batch_size=64, 
    shuffle=True
)

nn_model = CustomCyrillicNet().to(computing_device)

nn_model.load_state_dict(torch.load(weights_file, map_location=computing_device))


nn_model.eval()
acc_correct, acc_total = 0, 0


with torch.no_grad():
    for batch_index, (pics, targets) in enumerate(eval_dataloader):
        pics, targets = pics.to(computing_device), targets.to(computing_device)
        
        model_preds = nn_model(pics).argmax(1)
        if batch_index == 0:
            for k in range(min(12, len(pics))):
                pred_label = idx_to_str[model_preds[k].item()]
                real_label = idx_to_str[targets[k].item()]
                

                print(f"Expected : {real_label}")
                print(f"Predicted: {pred_label}")
                
        

        acc_total += targets.size(0)
        acc_correct += (model_preds == targets).sum().item()

print(f"\nAccuracy: {100 * acc_correct / acc_total:.2f}%")