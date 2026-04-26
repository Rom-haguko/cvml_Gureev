from pathlib import Path
import time

import cv2
from ultralytics import YOLO


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "figures" / "yolo" / "weights" / "best.pt"

model = YOLO(str(MODEL_PATH))

camera = cv2.VideoCapture(0)

cv2.namedWindow("YOLO detection", cv2.WINDOW_NORMAL)

enabled = True
last_boxes = []
missed_frames = 0
smooth = 0.7

while True:
    ok, frame = camera.read()

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("p"):
        enabled = not enabled
        print("Распознавание:", enabled)

    if enabled:
        start = time.time()
        results = model.predict(frame, conf=0.30, iou=0.45, verbose=False)
        work_time = time.time() - start

        current_boxes = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                object_name = model.names[class_id]

                if object_name == "neither":
                    continue

                current_boxes.append((x1, y1, x2, y2, object_name, confidence))

                if confidence >= 0.6:
                    print(
                        f"time={work_time:.3f}s, "
                        f"class={object_name}, "
                        f"conf={confidence:.2f}"
                    )

        if current_boxes:
            if last_boxes:
                smoothed_boxes = []

                for i, box in enumerate(current_boxes):
                    if i < len(last_boxes):
                        old_x1, old_y1, old_x2, old_y2, _, _ = last_boxes[i]
                        x1, y1, x2, y2, object_name, confidence = box

                        x1 = int(old_x1 * smooth + x1 * (1 - smooth))
                        y1 = int(old_y1 * smooth + y1 * (1 - smooth))
                        x2 = int(old_x2 * smooth + x2 * (1 - smooth))
                        y2 = int(old_y2 * smooth + y2 * (1 - smooth))

                        smoothed_boxes.append((x1, y1, x2, y2, object_name, confidence))
                    else:
                        smoothed_boxes.append(box)

                last_boxes = smoothed_boxes
            else:
                last_boxes = current_boxes

            missed_frames = 0
        else:
            missed_frames += 1

        if missed_frames < 5:
            for x1, y1, x2, y2, object_name, confidence in last_boxes:
                text = f"{object_name}: {confidence:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    frame,
                    text,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("YOLO detection", frame)

camera.release()
cv2.destroyAllWindows()