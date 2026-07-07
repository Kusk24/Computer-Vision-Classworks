# Name - Win Yu Maung
# ID - 6612054
# Sec - 541 

# Chapter 6 Recognition
# Classification, Object Detection, and Semantic Segmentation
 
# The provided Python examples use the OpenCV's dnn module to run pre-trained deep learning models.
 
# 1. Image Classification
# This Python code classifies an entire image using a pre-trained GoogLeNet model (from Caffe).
 
import cv2
import numpy as np
import os

# Resolve file paths relative to this script, so it works from any directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 1. Download required model files ---
# 1.1. Model architecture: bvlc_googlenet.prototxt
# 1.2. Pre-trained weights: bvlc_googlenet.caffemodel
# 1.3. Class labels: classification_classes_ILSVRC2012.txt
# Place these files in your project directory.
 
# --- 2. Load the model and labels ---
net = cv2.dnn.readNetFromCaffe(
    os.path.join(BASE_DIR, 'bvlc_googlenet.prototxt'),
    os.path.join(BASE_DIR, 'bvlc_googlenet.caffemodel'))
 
# Load class labels
with open(os.path.join(BASE_DIR, 'classification_classes_ILSVRC2012.txt'), 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
 
# --- 3. Load and pre-process an image ---
image = cv2.imread(os.path.join(BASE_DIR, 'your_image.jpg'))
 
# Create a blob (pre-processed input for the network)
# Resizes to 224x224, subtracts mean, and swaps channels (BGR -> RGB)
blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), (104, 117, 123))
 
# --- 4. Perform inference ---
net.setInput(blob)
output = net.forward()  # Output is a 1x1000 vector of probabilities
 
# --- 5. Interpret results ---
class_id = np.argmax(output)  # Index of the highest probability
confidence = output[0, class_id]
 
# Print the result
print(f"Predicted Class: {classes[class_id]}, Confidence: {confidence:.4f}")