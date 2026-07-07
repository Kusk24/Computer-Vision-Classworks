# Name - Win Yu Maung
# ID - 6612054
# Sec - 541 

# 3. Semantic Segmentation
# This Python code performs pixel-wise classification to segment an image using a DeepLabV3 model.
 
import cv2
import numpy as np
import os

# Resolve file paths relative to this script, so it works from any directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 1. Download required model files ---
# 1.1. Pre-trained segmentation model (FCN-ResNet50, ONNX) that outputs the 21
#       PASCAL VOC classes. OpenCV's DNN loads this cleanly, unlike raw DeepLab
#       TensorFlow graphs which fail to import (Assert/Reshape nodes).
# 1.2. Class labels and colors: pascal-classes.txt
# Place these files in your project directory.

# --- 2. Load the model and labels ---
net = cv2.dnn.readNetFromONNX(os.path.join(BASE_DIR, 'fcn-resnet50.onnx'))
 
# Load PASCAL VOC class labels and their associated colors
classes = []
colors = []
with open(os.path.join(BASE_DIR, 'pascal-classes.txt'), 'rt') as f:
    for line in f:
        # Example line: "background 0 0 0"
        parts = line.strip().split(' ')
        classes.append(parts[0])
        colors.append([int(c) for c in parts[1:]])
 
# --- 3. Load and pre-process an image ---
image = cv2.imread(os.path.join(BASE_DIR, 'your_image.jpg'))
height, width = image.shape[:2]

# FCN-ResNet50 expects a 513x513 RGB input, scaled to [0,1] and normalized with
# the ImageNet mean/std.
blob = cv2.dnn.blobFromImage(image, 1.0 / 255, (513, 513),
                             (0.485 * 255, 0.456 * 255, 0.406 * 255), swapRB=True)
blob[0, 0] /= 0.229
blob[0, 1] /= 0.224
blob[0, 2] /= 0.225
 
# --- 4. Perform inference ---
net.setInput(blob)
output = net.forward()  # Shape: (1, 21, 513, 513) for 21 PASCAL classes
 
# --- 5. Process the output to create a colored segmentation mask ---
# Get the class with the highest probability for each pixel
# Shape becomes (513, 513) with class IDs
output = np.argmax(output[0], axis=0)
 
# Resize the mask back to the original image size using nearest neighbor
mask = cv2.resize(output.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
 
# Convert the class IDs to their corresponding colors
colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
for class_id, color in enumerate(colors):
    colored_mask[mask == class_id] = color
 
# Overlay the mask on the original image
result = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
 
cv2.imshow('Semantic Segmentation', result)
cv2.waitKey(0)
cv2.destroyAllWindows()