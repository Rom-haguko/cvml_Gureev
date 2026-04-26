from pathlib import Path
import yaml
import torch
from ultralytics import YOLO


BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "spheres_and_cubes_new"

CLASS_NAMES = {
    0: "cube",
    1: "neither",
    2: "sphere"
}


def make_dataset_config():
    data = {
        "path": str(DATASET_DIR.resolve()),
        "train": str((DATASET_DIR / "images" / "train").resolve()),
        "val": str((DATASET_DIR / "images" / "val").resolve()),
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES
    }

    config_path = DATASET_DIR / "dataset.yaml"

    with open(config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True)

    return config_path


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    yaml_path = make_dataset_config()

    model = YOLO("yolo26n.pt")

    result = model.train(
        data=str(yaml_path),
        imgsz=640,
        epochs=50,
        batch=16,
        workers=2,

        optimizer="AdamW",
        lr0=0.001,
        patience=5,
        warmup_epochs=5,
        cos_lr=True,

        dropout=0.3,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        degrees=5.0,
        scale=0.5,
        translate=0.1,

        conf=0.01,
        iou=0.7,

        project=BASE_DIR / "figures",
        name="yolo",
        save=True,
        save_period=5,
        device=get_device(),

        verbose=True,
        plots=True,
        val=True,
        close_mosaic=8,
        amp=False
    )

    print("Training finished")
    print(result.save_dir)