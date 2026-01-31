import os
import uuid
import cv2
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from inference.detect_faces import detect_faces_bgr
from inference.predict_classifier import predict_faces
from inference.visualize import draw_boxes

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "outputs")

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

MODEL_PATHS = {
    "cnn": os.path.join(os.path.dirname(BASE_DIR), "cnn_face_classifier.pth"),
    "resnet18": os.path.join(os.path.dirname(BASE_DIR), "resnet_face_classifier.pth"),
}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["OUTPUT_FOLDER"] = OUTPUT_DIR


def _allowed_ext(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_IMAGE_EXT or ext in ALLOWED_VIDEO_EXT


def _ensure_dirs():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _summarize(preds):
    real_count = sum(1 for p in preds if p["label"] == "real")
    fake_count = sum(1 for p in preds if p["label"] == "fake")
    return {"real": real_count, "fake": fake_count, "total": len(preds)}


def process_image(image_path, model_name):
    image = cv2.imread(image_path)
    if image is None:
        return None, None, "Could not read the image."

    boxes = detect_faces_bgr(image)
    if not boxes:
        return None, None, "No faces detected."

    face_crops = [image[y:y + h, x:x + w] for (x, y, w, h) in boxes]
    preds = predict_faces(face_crops, model_name=model_name, model_path=MODEL_PATHS[model_name])

    output = draw_boxes(image, boxes, preds)
    out_name = f"image_{uuid.uuid4().hex}.jpg"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, output)

    return out_name, _summarize(preds), None


def process_video(video_path, model_name, frame_stride=2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, "Could not open the video."

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_name = f"video_{uuid.uuid4().hex}.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_idx = 0
    all_preds = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride != 0:
            writer.write(frame)
            frame_idx += 1
            continue

        boxes = detect_faces_bgr(frame)
        if boxes:
            face_crops = [frame[y:y + h, x:x + w] for (x, y, w, h) in boxes]
            preds = predict_faces(face_crops, model_name=model_name, model_path=MODEL_PATHS[model_name])
            frame = draw_boxes(frame, boxes, preds)
            all_preds.extend(preds)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    summary = _summarize(all_preds) if all_preds else {"real": 0, "fake": 0, "total": 0}
    return out_name, summary, None


@app.route("/", methods=["GET", "POST"])
def index():
    _ensure_dirs()
    error = None
    image_url = None
    video_url = None
    summary = None

    if request.method == "POST":
        file = request.files.get("file")
        model_name = request.form.get("model", "cnn")

        if not file or file.filename == "":
            error = "Please choose a file."
        elif model_name not in MODEL_PATHS:
            error = "Invalid model selection."
        elif not os.path.exists(MODEL_PATHS[model_name]):
            error = f"Model weights not found: {MODEL_PATHS[model_name]}"
        elif not _allowed_ext(file.filename):
            error = "Unsupported file type."
        else:
            filename = secure_filename(file.filename)
            file_ext = os.path.splitext(filename)[1].lower()
            unique_name = f"upload_{uuid.uuid4().hex}{file_ext}"
            upload_path = os.path.join(UPLOAD_DIR, unique_name)
            file.save(upload_path)

            if file_ext in ALLOWED_IMAGE_EXT:
                out_name, summary, error = process_image(upload_path, model_name)
                if out_name:
                    image_url = f"outputs/{out_name}"
            else:
                out_name, summary, error = process_video(upload_path, model_name)
                if out_name:
                    video_url = f"outputs/{out_name}"

    return render_template("index.html",
                           error=error,
                           image_url=image_url,
                           video_url=video_url,
                           summary=summary)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
