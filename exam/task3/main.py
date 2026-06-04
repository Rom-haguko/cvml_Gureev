import cv2
import numpy as np
from pathlib import Path
from train import BoxDetector
import torch
from torchvision import transforms

model_path = Path(__file__).parent / "bbox_model.pth"

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

model = BoxDetector().to("mps")
model.load_state_dict(torch.load(model_path, map_location="mps"))
model.eval()

canvas = np.zeros((256, 256), dtype="uint8")

cv2.namedWindow("Canvas", cv2.WINDOW_GUI_NORMAL)

position = []
draw = False


def on_mouse(event, x, y, flags, param):
    global draw
    global position

    if event == cv2.EVENT_LBUTTONDOWN:
        draw = True

    if event == cv2.EVENT_LBUTTONUP:
        draw = False

    if event == cv2.EVENT_MOUSEMOVE and draw:
        position = [y, x]


cv2.setMouseCallback("Canvas", on_mouse)

while True:
    if position:
        cv2.circle(canvas, (position[1], position[0]), 5, 255, -1)

    key = cv2.waitKey(1) & 0xFF

    match key:
        case 27:
            break
        case 99:
            position = []
            canvas *= 0

    if canvas.any():
        with torch.no_grad():
            tensor = transform(canvas)
            batch = tensor.unsqueeze(0).to("mps")

            bbox = model(batch)
            bbox = bbox.squeeze().cpu().numpy()

            x1 = int(bbox[0] * 256)
            y1 = int(bbox[1] * 256)
            x2 = int(bbox[2] * 256)
            y2 = int(bbox[3] * 256)

            print(x1, y1, x2, y2)

        box = canvas.copy()

        cv2.rectangle(
            box,
            (x1, y1),
            (x2, y2),
            255,
            1
        )

        cv2.imshow("Canvas", box)

    else:
        cv2.imshow("Canvas", canvas)

cv2.destroyAllWindows()