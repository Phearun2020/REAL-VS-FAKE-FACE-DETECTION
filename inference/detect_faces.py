import cv2

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_FACE_DETECTOR = cv2.CascadeClassifier(CASCADE_PATH)


def detect_faces_bgr(image_bgr, scale_factor=1.1, min_neighbors=5, min_size=(50, 50)):
    """
    Detect faces in a BGR image using Haar cascades.
    Returns list of (x, y, w, h).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = _FACE_DETECTOR.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size
    )

    if len(faces) == 0:
        return []

    # Sort by area (largest first)
    faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
    return faces
