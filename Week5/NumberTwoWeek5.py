# Name - Win Yu Maung
# ID - 6612054
# Sec - 541 

# 2. Object Detection
# This Python code detects and localizes objects in an image using an SSD (Single Shot Detector) model with a MobileNet backbone.
 
# Note: In computer vision, COCO (Common Objects in Context) refers to a massive, widely-used dataset and standardized annotation format created by Microsoft. It serves as the primary industry benchmark for training and evaluating machine learning models on tasks like object detection, image segmentation, and captioning.
 
import cv2
import os

# Resolve file paths relative to this script, so it works from any directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 1. Download required model files ---
# 1.1. Model configuration: ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
# 1.2. Pre-trained weights: frozen_inference_graph.pb
# 1.3. Class labels: coco.names
# Place these files in your project directory.
 
# --- 2. Load the model and labels ---
# A more convenient way to load detection models
model = cv2.dnn_DetectionModel(
    os.path.join(BASE_DIR, 'frozen_inference_graph.pb'),
    os.path.join(BASE_DIR, 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'))
 
# Load COCO class names
with open(os.path.join(BASE_DIR, 'coco.names'), 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
 
# --- 3. Configure input parameters ---
model.setInputSize(320, 320)          # Input image size
model.setInputScale(1.0 / 127.5)      # Scaling factor
model.setInputMean((127.5, 127.5, 127.5)) # Mean subtraction
model.setInputSwapRB(True)            # Swap channels (BGR -> RGB) [citation:4]
 
# --- 4. Load an image and detect objects ---
image = cv2.imread(os.path.join(BASE_DIR, 'your_image.jpg'))
class_ids, confidences, bounding_boxes = model.detect(image, confThreshold=0.5)
 
# --- 5. Visualize the results ---
for class_id, confidence, bbox in zip(class_ids.flatten(), confidences.flatten(), bounding_boxes):
    # Draw bounding box
    cv2.rectangle(image, bbox, color=(0, 255, 0), thickness=2)
 
    # Draw label and confidence
    label = f"{classes[class_id - 1]}: {confidence:.2f}"  # class_id is 1-indexed
    cv2.putText(image, label, (bbox[0] + 10, bbox[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
 
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()