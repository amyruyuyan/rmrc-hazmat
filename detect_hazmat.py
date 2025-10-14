import collections
import torch
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
import cv2
import numpy as np

# Set device for Mac (Apple Silicon)
torch.set_default_device(torch.device("mps"))

# Load class names from Hazmat_Individual subfolders
import os
DATA_PATH = "Hazmat_Individual"
class_names = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir
(os.path.join(DATA_PATH, d))])

# Load model
weights = ResNet50_Weights.DEFAULT
res = resnet50(weights=weights)
res.fc = torch.nn.Identity()
for param in res.parameters():
    param.requires_grad = False
model = torch.nn.Sequential(collections.OrderedDict([
    ('resnet', res),
    ('final', torch.nn.Linear(in_features=2048, out_features=len(class_names))),
    ('softmax', torch.nn.Softmax(dim=1)),
]))
model.load_state_dict(torch.load("hazmat_weights_individual.pth", 
map_location="mps"))
model.eval()

# Get preprocessing transform   
transform = torchvision.transforms.Compose([
    weights.transforms(),
])

def preprocess(frame):
    # Convert BGR (OpenCV) to RGB (PyTorch)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torchvision.transforms.ToPILImage()(img)
    img = transform(img)
    return img.unsqueeze(0)  # Add batch dimension

# OpenCV video capture
cv2.namedWindow('Hazmat Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hazmat Detection', 800, 600)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break
    # Preprocess and predict
    input_tensor = preprocess(frame).to("mps")
    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(1).item()
        pred_label = class_names[pred_idx]
        confidence = output[0, pred_idx].item()
    # Display prediction
    label_text = f"{pred_label} ({confidence*100:.1f}%)"
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
    (0,255,0), 2)
    cv2.imshow('Hazmat Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
