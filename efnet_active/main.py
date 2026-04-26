import cv2
import torch
import torchvision
from torchvision import transforms
from pathlib import Path
import time

BASE_DIR = Path(__file__).parent
WEIGHTS_PATH = BASE_DIR / "model.pth"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def load_inference_model(weights_path):
    model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 1)
    
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print("Model loaded.")
    else:
        print("WARNING: model.pth not found!")
        
    return model.to(device).eval()

model = load_inference_model(WEIGHTS_PATH)

img_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = img_transforms(img_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(tensor).squeeze(-1)
        prob = torch.sigmoid(out).item()
        
    label = "Person" if prob > 0.5 else "No Person"
    return label, prob

cap = cv2.VideoCapture(0)
cv2.namedWindow("Inference", cv2.WINDOW_GUI_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    cv2.imshow("Inference", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
    elif key == ord("p"):
        start_time = time.time()
        label, conf = predict_frame(frame)
        
        print(f"\nTime: {time.time() - start_time:.4f}s")
        print(f"Result: {label} (Conf: {conf:.3f})")
        print("-" * 30)

cap.release()
cv2.destroyAllWindows()
