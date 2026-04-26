import cv2
import torch
import torchvision
from torchvision import transforms
from collections import deque
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent
WEIGHTS_PATH = BASE_DIR / "model.pth"

device = torch.device("mps")


def create_efnet_model(weights_path):
    model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    for param in model.features.parameters():
        param.requires_grad = False
        
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 1)
    
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        
    return model.to(device)

model = create_efnet_model(WEIGHTS_PATH)
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

img_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class FrameBuffer:
    def __init__(self, capacity=16):
        self.capacity = capacity
        self.images = deque(maxlen=capacity)
        self.targets = deque(maxlen=capacity)

    def add_data(self, img_tensor, label):
        self.images.append(img_tensor)
        self.targets.append(label)

    def get_tensors(self):
        x = torch.stack(list(self.images))
        y = torch.tensor(list(self.targets), dtype=torch.float32)
        return x, y

def train_step(data_buffer):
    if len(data_buffer.images) < data_buffer.capacity:
        return None
    
    model.train()
    x, y = data_buffer.get_tensors()
    x, y = x.to(device), y.to(device) 
    
    optimizer.zero_grad()
    preds = model(x).squeeze(-1)
    loss = loss_fn(preds, y)
    loss.backward()
    optimizer.step()
    
    return loss.item()

def make_prediction(frame):
    model.eval()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = img_transforms(img_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(tensor).squeeze(-1)
        probability = torch.sigmoid(out).item()
        
    res_label = "Person" if probability > 0.5 else "No Person"
    return res_label, probability

cap = cv2.VideoCapture(0)
cv2.namedWindow("Active Learning", cv2.WINDOW_GUI_NORMAL)
buffer = FrameBuffer(capacity=16)
labels_collected = 0

print("Keys: '1'-Person, '2'-No Person, 'p'-Predict, 's'-Save, 'q'-Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Active Learning", frame)
    key = cv2.waitKey(1) & 0xFF

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if key == ord("q"):
        break
    elif key == ord("1"):
        buffer.add_data(img_transforms(img_rgb), 1.0)
        labels_collected += 1
        print(f"Added: Person. Buffer: {labels_collected}/{buffer.capacity}")
    elif key == ord("2"):
        buffer.add_data(img_transforms(img_rgb), 0.0)
        labels_collected += 1
        print(f"Added: No Person. Buffer: {labels_collected}/{buffer.capacity}")
    elif key == ord("p"):
        start_t = time.time()
        label, conf = make_prediction(frame)
        print(f"[{time.time() - start_t:.3f}s] {label} (Conf: {conf:.2f})")
    elif key == ord("s"):
        torch.save(model.state_dict(), WEIGHTS_PATH)
        print("Weights saved!")

    if labels_collected >= buffer.capacity:
        loss_val = train_step(buffer)
        if loss_val is not None:
            print(f"Train step done. Loss: {loss_val:.4f}")
            with open(BASE_DIR / "loss_log.txt", "a", encoding="utf-8") as f:
                f.write(f"Loss EffNet = {loss_val:.4f}\n")
        labels_collected = 0 

cap.release()
cv2.destroyAllWindows()
