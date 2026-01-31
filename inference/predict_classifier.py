import torch
import torch.nn.functional as F
from PIL import Image
import cv2

from models.cnn_classifier import CNNClassifier
from models.resnet_classifier import build_resnet18
from utils.dataset import get_transforms

_MODEL_CACHE = {}


def _load_model(model_name, model_path, device):
    cache_key = (model_name, model_path, str(device))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if model_name == "cnn":
        model = CNNClassifier(num_classes=2)
    elif model_name == "resnet18":
        model = build_resnet18(num_classes=2, pretrained=False)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    _MODEL_CACHE[cache_key] = model
    return model


def predict_faces(face_images_bgr, model_name="cnn", model_path="cnn_face_classifier.pth", device=None):
    """
    Predict real/fake for a list of BGR face crops.
    Returns list of dicts: {label, prob_real, prob_fake}.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = _load_model(model_name, model_path, device)
    transform = get_transforms(train=False)

    results = []
    for face_bgr in face_images_bgr:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)
        tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        prob_real = float(probs[0])
        prob_fake = float(probs[1])
        label = "real" if prob_real >= prob_fake else "fake"

        results.append({
            "label": label,
            "prob_real": prob_real,
            "prob_fake": prob_fake
        })

    return results
