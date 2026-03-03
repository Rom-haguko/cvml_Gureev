import cv2
import numpy as np
from pathlib import Path
from skimage.io import imread
from skimage.measure import label, regionprops

data = Path("task")
train_path = data / "train"

def extractor(image):
    p = max(regionprops(label(image)), key=lambda x: x.area)
    h, w = p.image.shape
    cy, cx = p.centroid_local
    return np.array([p.eccentricity, p.solidity, p.extent, p.euler_number, p.perimeter / p.area, cy / h, cx / w], np.float32)

def make_train(path):
    train, responses, letters, ncls = [], [], {}, 0
    for cls in sorted(path.glob("*")):
        if cls.is_dir():
            ncls += 1
            name = cls.name[1:] if cls.name.startswith("s") and len(cls.name) > 1 else cls.name
            letters[ncls] = name
            for p in cls.glob("*.png"):
                train.append(extractor(imread(p).mean(2) > 4))
                responses.append(ncls)
    return np.array(train, np.float32), np.array(responses, np.float32).reshape(-1, 1), letters

def merge_boxes(props, eps):
    m = []
    for p in props:
        y1, x1, y2, x2 = p.bbox
        cx = p.centroid[1]
        if m and abs(cx - m[-1][4]) < eps:
            a, b, c, d, lcx = m[-1]
            m[-1] = (min(y1, a), min(x1, b), max(y2, c), max(x2, d), lcx)
        else:
            m.append((y1, x1, y2, x2, cx))
    return [(a, b, c, d) for a, b, c, d, _ in m]

def space_thr(boxes):
    g = np.array([boxes[i][1] - boxes[i-1][3] for i in range(1, len(boxes))], np.float32)
    return float(np.median(g) + 2.5 * (np.percentile(g, 75) - np.percentile(g, 25)))

train, responses, letters = make_train(train_path)
knn = cv2.ml.KNearest.create()
knn.train(train, cv2.ml.ROW_SAMPLE, responses)

for i in range(7):
    binary = imread(data / f"{i}.png").mean(2) > 4
    props = sorted(regionprops(label(binary)), key=lambda p: p.bbox[1])
    widths = np.array([p.bbox[3] - p.bbox[1] for p in props], np.float32)
    boxes = merge_boxes(props, eps=float(np.median(widths)) * 0.5)
    thr = space_thr(boxes)

    res, x2_p = "", None
    for y1, x1, y2, x2 in boxes:
        if x2_p is not None and (x1 - x2_p) > thr:
            res += " "
        _, r, _, _ = knn.findNearest(extractor(binary[y1:y2, x1:x2])[None, :], 5)
        res += letters[int(r[0, 0])]
        x2_p = x2
    print(res)
