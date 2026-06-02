from pathlib import Path

import cv2


def draw_yolo_pose(img_path, label_path):
    img = cv2.imread(str(img_path))
    h, w, _ = img.shape

    if not label_path.exists():
        print(f"Label not found for {img_path.name}")
        return img

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        data = list(map(float, line.split()))
        # YOLO Pose format: [class, x_c, y_c, w, h, px1, py1, v1, px2, py2, v2, ...]
        # Перші 5 значень — це BBox, далі йдуть точки (по 3 значення на кожну)
        keypoints = data[5:]

        for i in range(0, len(keypoints), 3):
            px = int(keypoints[i] * w)
            py = int(keypoints[i + 1] * h)
            visibility = int(keypoints[i + 2])

            # Колір залежно від видимості:
            # 2 - видима (Green), 1 - перекрита (Yellow), 0 - за кадром (Red)
            color = (0, 255, 0) if visibility == 2 else (0, 255, 255) if visibility == 1 else (0, 0, 255)

            if visibility > 0:
                cv2.circle(img, (px, py), 4, color, -1)
                cv2.putText(img, str(i // 3), (px + 5, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return img


root = Path(r"Q:\Projects\KHNU\PointsX\data\synthetic-pose\train")
sample_img = next((root / "images").glob("*.jpg"))
sample_lbl = root / "labels" / f"{sample_img.stem}.txt"

result = draw_yolo_pose(sample_img, sample_lbl)
cv2.imshow("Keypoint Verification", result)
cv2.waitKey(0)
cv2.destroyAllWindows()