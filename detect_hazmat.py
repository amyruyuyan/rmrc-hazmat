import collections
import torch
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
import cv2
import numpy as np

'''
add: function that returns the string of what the label is 
- if there is nothing return empty list
function should also draw the bounding box + return the frame
the bounding box should be a green box with the labl on top (no score confidence)

might have to adjust the pathing for the model and data
'''

# Set device for Mac (Apple Silicon)
torch.set_default_device(torch.device("mps"))

# Load class names from Hazmat_Individual subfolders
import os
DATA_PATH = "hazmatstuff/Hazmat_Individual"
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
model.load_state_dict(torch.load("hazmatstuff/hazmat_weights_individual.pth", 
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

def find_hazmat_diamond(frame):
    """
    Detect diamond-shaped hazmat signs in the frame using contour detection.
    
    Args:
        frame: Input image frame (OpenCV format - BGR)
        
    Returns:
        list: List of bounding boxes [(x, y, w, h)] for detected diamonds
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive threshold to handle varying lighting
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    diamond_boxes = []
    
    for contour in contours:
        # Filter by area (hazmat signs should be reasonably sized)
        area = cv2.contourArea(contour)
        if area < 500 or area > frame.shape[0] * frame.shape[1] * 0.5:
            continue
        
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if it's roughly diamond-shaped (4 vertices)
        if len(approx) >= 4:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio (diamonds are roughly square)
            aspect_ratio = float(w) / h
            if 0.7 <= aspect_ratio <= 1.4:
                # Check if contour is convex (diamonds should be)
                if cv2.isContourConvex(approx):
                    diamond_boxes.append((x, y, w, h))
    
    return diamond_boxes

def detect_hazmat(frame, confidence_threshold=0.4):
    """
    Detects hazmat in a frame and draws bounding box with label.
    
    Args:
        frame: Input image frame (OpenCV format - BGR)
        confidence_threshold: Minimum confidence to consider detection valid
        
    Returns:
        tuple: (detected_labels, annotated_frame)
            - detected_labels: list of detected label strings, empty list if nothing detected
            - annotated_frame: frame with green bounding box and label drawn
    """
    # Preprocess and predict
    input_tensor = preprocess(frame).to("mps")
    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(1).item()
        pred_label = class_names[pred_idx]
        confidence = output[0, pred_idx].item()
    
    # Copy frame to avoid modifying original
    annotated_frame = frame.copy()
    detected_labels = []
    
    # Check if confidence meets threshold
    if confidence >= confidence_threshold:
        detected_labels.append(pred_label)
        
        # Try to find actual hazmat diamond shapes
        diamond_boxes = find_hazmat_diamond(frame)
        
        # If we found diamond shapes, use the largest one
        if diamond_boxes:
            # Sort by area and take the largest
            diamond_boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
            x, y, w, h = diamond_boxes[0]
        else:
            # Fallback: use center area if no diamond detected
            height, width = frame.shape[:2]
            box_width = int(width * 0.3)
            box_height = int(height * 0.3)
            center_x, center_y = width // 2, height // 2
            x = center_x - box_width // 2
            y = center_y - box_height // 2
            w, h = box_width, box_height
        
        # Draw green bounding box around the detected hazmat area
        cv2.rectangle(annotated_frame, 
                     (x, y), 
                     (x + w, y + h), 
                     (0, 255, 0), 3)
        
        # Draw label on top of bounding box
        label_text = pred_label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        # Get text size to position it properly
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        # Draw text background rectangle above the bounding box
        cv2.rectangle(annotated_frame,
                     (x, y - text_height - 10),
                     (x + text_width + 10, y),
                     (0, 255, 0), -1)
        
        # Draw text
        cv2.putText(annotated_frame, label_text, 
                   (x + 5, y - 5), 
                   font, font_scale, (0, 0, 0), thickness)
    
    return detected_labels, annotated_frame

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
    
    # Use the new detect_hazmat function
    detected_labels, annotated_frame = detect_hazmat(frame)
    
    # Display the annotated frame
    cv2.imshow('Hazmat Detection', annotated_frame)
    
    # Print detected labels (optional)
    if detected_labels:
        print(f"Detected: {detected_labels}")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
