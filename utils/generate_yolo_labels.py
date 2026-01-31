import cv2
import os

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(CASCADE_PATH)

# Change paths if needed
BASE_CLASSIFICATION_DIR = "data/classification"
BASE_DETECTION_DIR = "data/detection"

SPLITS = ["train", "val", "test"]
CLASSES = {"real": 0, "fake": 1}


def convert_to_yolo(x, y, w, h, img_w, img_h):
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return x_center, y_center, w_norm, h_norm


for split in SPLITS:
    for class_name, class_id in CLASSES.items():

        img_dir = os.path.join(BASE_CLASSIFICATION_DIR, split, class_name)
        out_img_dir = os.path.join(BASE_DETECTION_DIR, "images", split)
        out_label_dir = os.path.join(BASE_DETECTION_DIR, "labels", split)

        os.makedirs(out_label_dir, exist_ok=True)

        for img_name in os.listdir(img_dir):
            if not img_name.lower().endswith((".jpg", ".png")):
                continue

            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            h, w, _ = img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5
            )

            if len(faces) == 0:
                continue  # skip images with no detected face

            # Take largest detected face
            x, y, bw, bh = max(faces, key=lambda b: b[2] * b[3])

            xc, yc, bw_n, bh_n = convert_to_yolo(x, y, bw, bh, w, h)

            label_path = os.path.join(
                out_label_dir, img_name.rsplit(".", 1)[0] + ".txt"
            )

            with open(label_path, "w") as f:
                f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw_n:.6f} {bh_n:.6f}\n")

print("YOLO label generation complete.")
