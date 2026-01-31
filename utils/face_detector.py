import os
import cv2
import torch

try:
    from facenet_pytorch import MTCNN
except Exception:
    MTCNN = None


_WARNED = False


def _get_backend():
    return os.environ.get("FACE_DETECTOR", "mtcnn").lower()


class FaceDetector:
    def __init__(self, backend="mtcnn", min_confidence=0.90, min_size=40):
        self.backend = backend
        self.min_confidence = min_confidence
        self.min_size = min_size

        if backend == "mtcnn":
            if MTCNN is None:
                raise ImportError("facenet_pytorch is not installed")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.detector = MTCNN(keep_all=True, device=device, min_face_size=min_size)
        elif backend == "haar":
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.detector = cv2.CascadeClassifier(cascade_path)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def detect(self, image_bgr):
        if self.backend == "mtcnn":
            return self._detect_mtcnn(image_bgr)
        return self._detect_haar(image_bgr)

    def _detect_mtcnn(self, image_bgr):
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes, probs = self.detector.detect(image_rgb)

        if boxes is None or probs is None:
            return []

        results = []
        for box, prob in zip(boxes, probs):
            if prob is None or prob < self.min_confidence:
                continue
            x1, y1, x2, y2 = box
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w - 1, int(x2))
            y2 = min(h - 1, int(y2))
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            if bw < self.min_size or bh < self.min_size:
                continue
            results.append((x1, y1, bw, bh))

        results.sort(key=lambda b: b[2] * b[3], reverse=True)
        return results

    def _detect_haar(self, image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_size, self.min_size)
        )
        if len(faces) == 0:
            return []
        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        return faces


def detect_faces_bgr(image_bgr, min_confidence=0.90, min_size=40):
    global _WARNED
    backend = _get_backend()

    if backend == "mtcnn":
        try:
            detector = FaceDetector(backend="mtcnn", min_confidence=min_confidence, min_size=min_size)
            return detector.detect(image_bgr)
        except Exception:
            if not _WARNED:
                print("MTCNN unavailable; falling back to Haar cascade.")
                _WARNED = True
            backend = "haar"

    detector = FaceDetector(backend="haar", min_confidence=min_confidence, min_size=min_size)
    return detector.detect(image_bgr)
