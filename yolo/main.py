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

while True:
    ok, frame = camera.read()

    if not ok:
        print("Не удалось получить кадр")
        break

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("p"):
        enabled = not enabled
        print("Распознавание:", enabled)

    if enabled:
        start = time.time()
        results = model.predict(frame, conf=0.4, verbose=False)
        work_time = time.time() - start

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                object_name = model.names[class_id]

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

                print(
                    f"time={work_time:.3f}s, "
                    f"class={object_name}, "
                    f"conf={confidence:.2f}"
                )

    cv2.imshow("YOLO detection", frame)

camera.release()
cv2.destroyAllWindows()