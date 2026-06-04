import cv2
import numpy as np
from pathlib import Path
from train import LeNet5
import torch
from torchvision import transforms

model_path = Path(__file__).parent / "lenet5.pth"

classes = [
    "零", "一", "二", "三", "四",
    "五", "六", "七", "八", "九",
    "十", "百", "千", "万", "亿"
]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

model = LeNet5(num_classes=15).to("mps")
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
        cv2.circle(canvas, (position[1], position[0]), 3, 255, -1)

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

            output = model(batch)
            prediction = output.argmax(dim=1).item()

            probability = torch.softmax(output, dim=1).squeeze().cpu().numpy()
            confidence = (probability[prediction] * 100).astype("uint8")

            print(classes[prediction], confidence)

        text = canvas.copy()

        cv2.putText(
            text,
            f"{prediction + 1} {confidence}%",
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            255,
            1
        )

        cv2.imshow("Canvas", text)

    else:
        cv2.imshow("Canvas", canvas)

cv2.destroyAllWindows()