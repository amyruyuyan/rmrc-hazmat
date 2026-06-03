import collections
import os

import cv2
import torch
import torchvision
from torchvision.models import ResNet50_Weights, resnet50


_MODEL = None
_CLASS_NAMES = None
_DEVICE = None
_TRANSFORM = None


def _resolve_device(device_str):
    requested = (device_str or "cpu").lower()

    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    return torch.device("cpu")


def init_inference(data_path="hazmatstuff/Hazmat_Individual", weights_path="hazmatstuff/hazmat_weights_individual.pth", device_str="cpu"):
    global _MODEL, _CLASS_NAMES, _DEVICE, _TRANSFORM

    _DEVICE = _resolve_device(device_str)
    _CLASS_NAMES = sorted(
        [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    )

    weights = ResNet50_Weights.DEFAULT
    backbone = resnet50(weights=weights)
    backbone.fc = torch.nn.Identity()
    for param in backbone.parameters():
        param.requires_grad = False

    _MODEL = torch.nn.Sequential(
        collections.OrderedDict(
            [
                ("resnet", backbone),
                ("final", torch.nn.Linear(in_features=2048, out_features=len(_CLASS_NAMES))),
                ("softmax", torch.nn.Softmax(dim=1)),
            ]
        )
    )
    _MODEL.load_state_dict(torch.load(weights_path, map_location=_DEVICE))
    _MODEL.to(_DEVICE)
    _MODEL.eval()

    _TRANSFORM = weights.transforms()


def _preprocess(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = torchvision.transforms.ToPILImage()(image)
    image = _TRANSFORM(image)
    return image.unsqueeze(0)


def find_hazmat_diamond(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    diamond_boxes = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500 or area > frame.shape[0] * frame.shape[1] * 0.5:
            continue

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h else 0.0
        if 0.7 <= aspect_ratio <= 1.4:
            hull = cv2.convexHull(approx)
            hull_area = cv2.contourArea(hull)
            approx_area = cv2.contourArea(approx)
            if hull_area > 0 and (approx_area / hull_area) >= 0.9:
                diamond_boxes.append((x, y, w, h))

    return diamond_boxes


def _draw_label(annotated_frame, x, y, w, h, label_text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

    label_top_y = y - text_height - 10
    label_bottom_y = y
    text_org_y = y - 5

    if label_top_y < 0:
        label_top_y = min(y + h, annotated_frame.shape[0] - 1)
        label_bottom_y = min(y + h + text_height + 10, annotated_frame.shape[0] - 1)
        text_org_y = min(y + h + text_height + 5, annotated_frame.shape[0] - 1)

    cv2.rectangle(
        annotated_frame,
        (max(x, 0), max(label_top_y, 0)),
        (min(x + text_width + 10, annotated_frame.shape[1] - 1), max(label_bottom_y, 0)),
        (0, 255, 0),
        -1,
    )
    cv2.putText(
        annotated_frame,
        label_text,
        (max(x + 5, 0), max(text_org_y, 0)),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
    )


def run_frame(frame, confidence_threshold=0.4):
    if _MODEL is None or _CLASS_NAMES is None or _TRANSFORM is None:
        raise RuntimeError("Inference not initialized. Call init_inference() first.")

    input_tensor = _preprocess(frame).to(_DEVICE)
    with torch.no_grad():
        output = _MODEL(input_tensor)
        pred_idx = output.argmax(1).item()
        pred_label = _CLASS_NAMES[pred_idx]
        confidence = output[0, pred_idx].item()

    annotated_frame = frame.copy()
    detected_labels = []

    if confidence >= confidence_threshold:
        detected_labels.append(pred_label)
        diamond_boxes = find_hazmat_diamond(frame)

        if diamond_boxes:
            diamond_boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
            x, y, w, h = diamond_boxes[0]
        else:
            height, width = frame.shape[:2]
            box_width = int(width * 0.3)
            box_height = int(height * 0.3)
            center_x, center_y = width // 2, height // 2
            x = center_x - box_width // 2
            y = center_y - box_height // 2
            w, h = box_width, box_height

        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        _draw_label(annotated_frame, x, y, w, h, pred_label)

    return annotated_frame, detected_labels