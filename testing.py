import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os

print("Script started")

def get_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = get_model()

print("Loading weights...")
model.load_state_dict(torch.load("flir_person_detector_rgb_thermal.pth", map_location=device))
model.to(device)
model.eval()
print("Model loaded.")

if len(sys.argv) < 2:
    print("You forgot to pass an image path.")
    sys.exit()

image_path = sys.argv[1]
print("Image path:", image_path)

if not os.path.exists(image_path):
    print("Image file not found!")
    sys.exit()

img = Image.open(image_path).convert("RGB")
img_tensor = F.to_tensor(img).to(device)

print("Running inference...")
with torch.no_grad():
    prediction = model([img_tensor])[0]

print("Prediction done.")

boxes = prediction["boxes"]
scores = prediction["scores"]

print("Number of detections:", len(scores))

threshold = 0.5
human_detected = any(score > threshold for score in scores)
print("Human detected:", human_detected)

fig, ax = plt.subplots(1)
ax.imshow(img)

for box, score in zip(boxes, scores):
    if score > threshold:
        x1, y1, x2, y2 = box.cpu().numpy()
        
        # 1. Draw the Bounding Box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)

        # 2. Add the Confidence Score Label
        # We place it at (x1, y1 - 5) to sit slightly above the box
        label = f"Person: {score:.2f}"
        ax.text(
            x1, y1 - 5, label, 
            color='white', fontweight='bold',
            bbox=dict(facecolor='red', alpha=0.5, edgecolor='none')
        )
plt.show()