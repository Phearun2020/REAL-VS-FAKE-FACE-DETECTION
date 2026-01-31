import torch.nn as nn

try:
    from torchvision.models import resnet18, ResNet18_Weights
    _HAS_WEIGHTS_ENUM = True
except Exception:
    from torchvision.models import resnet18
    ResNet18_Weights = None
    _HAS_WEIGHTS_ENUM = False


def build_resnet18(num_classes=2, pretrained=False):
    """
    Build a ResNet18 classifier for real vs fake face detection.
    Output: logits for 2 classes (0: real, 1: fake).
    """
    if _HAS_WEIGHTS_ENUM:
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
    else:
        model = resnet18(pretrained=pretrained)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
