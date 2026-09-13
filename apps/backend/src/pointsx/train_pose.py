"""Finetune YOLO11n-pose on LV-MHP-v2 custom pose dataset."""

from pathlib import Path

from ultralytics import YOLO


def main():
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / "models" / "yolo11n-pose.pt"
    dataset_yaml = project_root / "data" / "LV-MHP-v2-pose" / "dataset.yaml"

    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {dataset_yaml}\n"
            "Run convert_pose.py first to generate the dataset."
        )

    model = YOLO(str(model_path))

    results = model.train(
        data=str(dataset_yaml),
        epochs=10,
        imgsz=640,
        batch=-1,  # auto batch size
        project=str(project_root / "runs"),
        name="pose",
        exist_ok=True,
        pretrained=True,
        patience=5,
        workers=4,
    )

    return results


if __name__ == "__main__":
    main()
