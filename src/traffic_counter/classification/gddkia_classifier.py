import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms

from traffic_counter.config import GDDKIA_LABELS, SUPPORTED_CLASSIFIER_ARCHS


def build_classifier_model(arch: str, num_classes: int):
    """
    Builds a torchvision model and replaces the classification head.

    Currently architecture names are expected to match the folder names from
    final_model_benchmark_09_06_2026, which is considered to be the final architectures list for the project.
    """

    arch = arch.lower().strip()

  # ---------------- ConvNeXt ----------------
    if arch == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model

    if arch == "convnext_small":
        model = models.convnext_small(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model

    if arch == "convnext_base":
        model = models.convnext_base(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model

    # ---------------- EfficientNetV2 ----------------
    if arch == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if arch == "efficientnet_v2_m":
        model = models.efficientnet_v2_m(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    # ---------------- Swin Transformer ----------------
    if arch == "swin_t":
        model = models.swin_t(weights=None)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    if arch == "swin_s":
        model = models.swin_s(weights=None)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    # ---------------- ResNet ----------------
    if arch == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if arch == "resnet101":
        model = models.resnet101(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # ---------------- RegNet ----------------
    if arch == "regnet_y_8gf":
        model = models.regnet_y_8gf(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # ---------------- DenseNet ----------------
    if arch == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    # ---------------- MobileNetV3 ----------------
    if arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model

    raise ValueError(
        f"Unsupported classifier architecture: {arch}. "
        f"Supported: {SUPPORTED_CLASSIFIER_ARCHS}"
    )

def load_model_state(model, model_path, device):
    """
    Load a model checkpoint robustly.

    Supports:
    - plain state_dict
    - checkpoint dict with 'model_state_dict'
    - state_dict saved from DataParallel with 'module.' prefix
    """
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Handle possible DataParallel prefix.
    if isinstance(state_dict, dict):
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                cleaned_state_dict[key[len("module."):]] = value
            else:
                cleaned_state_dict[key] = value
        state_dict = cleaned_state_dict

    model.load_state_dict(state_dict)
    return model

class GDDKIAClassifier:
    def __init__(self, model_path, device, arch: str = "convnext_small"):
        self.device = device
        self.arch = arch.lower().strip()

        self.model = build_classifier_model(arch, len(GDDKIA_LABELS))
        self.model = load_model_state(self.model, model_path, device)

        self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])

    @torch.inference_mode()
    def predict_index(self, crop_bgr):
        """
        Baseline: return only top class index.
        """
        probs = self.predict_proba(crop_bgr)
        return int(torch.argmax(probs).item())

    @torch.inference_mode()
    def predict_proba(self, crop_bgr):
        """
        Return full softmax probability vector for all GDDKiA classes.
        Change from BGR to RGB, apply transforms, and run through model.
        """
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        outputs = self.model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        return probs.squeeze(0).cpu()

    def label_from_index(self, idx: int) -> str:
        return GDDKIA_LABELS[idx]