import os
import uuid
import time
import json
import cv2
import numpy as np
from flask import Flask, render_template, request, Response

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except Exception:
    HEIF_AVAILABLE = False
from werkzeug.utils import secure_filename

from inference.detect_faces import detect_faces_bgr
from inference.predict_classifier import predict_faces
from inference.visualize import draw_boxes

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "outputs")

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".heic", ".heif"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

MODEL_PATHS = {
    "cnn": os.path.join(os.path.dirname(BASE_DIR), "cnn_face_classifier.pth"),
    "resnet18": os.path.join(os.path.dirname(BASE_DIR), "resnet_face_classifier.pth"),
}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["OUTPUT_FOLDER"] = OUTPUT_DIR

CAMERA_STATS = {
    "frames_analyzed": 0,
    "sum_confidence": 0.0,
    "num_predictions": 0,
    "started_at": None
}


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

def _avg_confidence(preds):
    if not preds:
        return 0.0
    return sum(max(p.get("prob_real", 0.0), p.get("prob_fake", 0.0)) for p in preds) / len(preds)

def _to_per_face(preds):
    per_face = []
    for p in preds:
        conf = max(p.get("prob_real", 0.0), p.get("prob_fake", 0.0))
        per_face.append({
            "label": p.get("label", "unknown"),
            "prob_real": p.get("prob_real", 0.0),
            "prob_fake": p.get("prob_fake", 0.0),
            "confidence": conf
        })
    return per_face

def _load_metrics():
    metrics = {}
    for name, path in [("cnn", "results/cnn/metrics.json"), ("resnet18", "results/resnet/metrics.json")]:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                metrics[name] = data
            except Exception:
                metrics[name] = None
        else:
            metrics[name] = None
    return metrics


def _read_image_any(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        return img
    if not PIL_AVAILABLE:
        return None
    try:
        pil_img = Image.open(image_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img
    except Exception:
        return None


def process_image(image_path, model_name, threshold):
    start = time.perf_counter()
    image = _read_image_any(image_path)
    if image is None:
        return None, None, None, None, None, "Could not read the image."

    boxes = detect_faces_bgr(image)
    if not boxes:
        return None, None, None, None, None, "No faces detected."

    face_crops = [image[y:y + h, x:x + w] for (x, y, w, h) in boxes]
    preds = predict_faces(face_crops, model_name=model_name, model_path=MODEL_PATHS[model_name])

    output = draw_boxes(image, boxes, preds, threshold=threshold)
    out_name = f"image_{uuid.uuid4().hex}.jpg"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, output)

    elapsed = time.perf_counter() - start
    analysis = {
        "frames_analyzed": 1,
        "processing_time": elapsed,
        "avg_confidence": _avg_confidence(preds)
    }
    per_face = _to_per_face(preds)
    return out_name, _summarize(preds, threshold), analysis, [], per_face, None


def process_video(video_path, model_name, threshold, frame_stride=2):
    start = time.perf_counter()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None, None, None, "Could not open the video."

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
    analyzed_frames = 0
    frame_samples = []
    last_frame_preds = []

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
            last_frame_preds = preds

        if len(frame_samples) < 8:
            sample_name = f"frame_{uuid.uuid4().hex}.jpg"
            sample_path = os.path.join(OUTPUT_DIR, sample_name)
            cv2.imwrite(sample_path, frame)
            frame_samples.append(f"outputs/{sample_name}")

        analyzed_frames += 1
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    summary = _summarize(all_preds, threshold) if all_preds else {"real": 0, "fake": 0, "uncertain": 0, "total": 0}
    elapsed = time.perf_counter() - start
    analysis = {
        "frames_analyzed": analyzed_frames,
        "processing_time": elapsed,
        "avg_confidence": _avg_confidence(all_preds)
    }
    per_face = _to_per_face(last_frame_preds)
    return out_name, summary, analysis, frame_samples, per_face, None


def generate_camera_stream(model_name, threshold, frame_stride=1):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    frame_idx = 0
    CAMERA_STATS["frames_analyzed"] = 0
    CAMERA_STATS["sum_confidence"] = 0.0
    CAMERA_STATS["num_predictions"] = 0
    CAMERA_STATS["started_at"] = time.perf_counter()
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
                    CAMERA_STATS["frames_analyzed"] += 1
                    CAMERA_STATS["sum_confidence"] += sum(max(p.get("prob_real", 0.0), p.get("prob_fake", 0.0)) for p in preds)
                    CAMERA_STATS["num_predictions"] += len(preds)

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
    analysis = None
    frames = []
    per_face = []
    metrics = _load_metrics()

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
                out_name, summary, analysis, frames, per_face, error = process_image(upload_path, model_name, threshold)
                if out_name:
                    image_url = f"outputs/{out_name}"
            else:
                out_name, summary, analysis, frames, per_face, error = process_video(upload_path, model_name, threshold)
                if out_name:
                    video_url = f"outputs/{out_name}"

    return render_template("index.html",
                           error=error,
                           image_url=image_url,
                           video_url=video_url,
                           summary=summary,
                           analysis=analysis,
                           frames=frames,
                           per_face=per_face,
                           metrics=metrics,
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


@app.route("/camera_stats")
def camera_stats():
    elapsed = 0.0
    if CAMERA_STATS["started_at"] is not None:
        elapsed = time.perf_counter() - CAMERA_STATS["started_at"]
    avg_conf = 0.0
    if CAMERA_STATS["num_predictions"] > 0:
        avg_conf = CAMERA_STATS["sum_confidence"] / CAMERA_STATS["num_predictions"]
    return {
        "frames_analyzed": CAMERA_STATS["frames_analyzed"],
        "processing_time": elapsed,
        "avg_confidence": avg_conf
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
