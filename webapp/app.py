import os
import uuid
import cv2
from flask import Flask, render_template, request, Response
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


def _summarize(preds, threshold):
    real_count = 0
    fake_count = 0
    uncertain_count = 0
    for p in preds:
        conf = max(p.get("prob_real", 0.0), p.get("prob_fake", 0.0))
        if conf < threshold:
            uncertain_count += 1
        elif p["label"] == "real":
            real_count += 1
        else:
            fake_count += 1
    return {
        "real": real_count,
        "fake": fake_count,
        "uncertain": uncertain_count,
        "total": len(preds)
    }


def process_image(image_path, model_name, threshold):
    image = cv2.imread(image_path)
    if image is None:
        return None, None, "Could not read the image."

    boxes = detect_faces_bgr(image)
    if not boxes:
        return None, None, "No faces detected."

    face_crops = [image[y:y + h, x:x + w] for (x, y, w, h) in boxes]
    preds = predict_faces(face_crops, model_name=model_name, model_path=MODEL_PATHS[model_name])

    output = draw_boxes(image, boxes, preds, threshold=threshold)
    out_name = f"image_{uuid.uuid4().hex}.jpg"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, output)

    return out_name, _summarize(preds, threshold), None


def process_video(video_path, model_name, threshold, frame_stride=2):
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
            frame = draw_boxes(frame, boxes, preds, threshold=threshold)
            all_preds.extend(preds)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    summary = _summarize(all_preds, threshold) if all_preds else {"real": 0, "fake": 0, "uncertain": 0, "total": 0}
    return out_name, summary, None


def generate_camera_stream(model_name, threshold, frame_stride=1):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride == 0:
                boxes = detect_faces_bgr(frame)
                if boxes:
                    face_crops = [frame[y:y + h, x:x + w] for (x, y, w, h) in boxes]
                    preds = predict_faces(face_crops, model_name=model_name, model_path=MODEL_PATHS[model_name])
                    frame = draw_boxes(frame, boxes, preds, threshold=threshold)

            frame_idx += 1

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    finally:
        cap.release()


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
        try:
            threshold = float(request.form.get("threshold", 0.5))
        except ValueError:
            threshold = 0.5

        threshold = max(0.0, min(1.0, threshold))

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
                out_name, summary, error = process_image(upload_path, model_name, threshold)
                if out_name:
                    image_url = f"outputs/{out_name}"
            else:
                out_name, summary, error = process_video(upload_path, model_name, threshold)
                if out_name:
                    video_url = f"outputs/{out_name}"

    return render_template("index.html",
                           error=error,
                           image_url=image_url,
                           video_url=video_url,
                           summary=summary,
                           threshold=threshold if request.method == "POST" else 0.5)


@app.route("/camera_feed")
def camera_feed():
    model_name = request.args.get("model", "cnn")
    try:
        threshold = float(request.args.get("threshold", 0.5))
    except ValueError:
        threshold = 0.5
    threshold = max(0.0, min(1.0, threshold))
    try:
        frame_stride = int(request.args.get("stride", 1))
    except ValueError:
        frame_stride = 1
    frame_stride = max(1, frame_stride)

    if model_name not in MODEL_PATHS:
        model_name = "cnn"

    return Response(generate_camera_stream(model_name, threshold, frame_stride=frame_stride),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
