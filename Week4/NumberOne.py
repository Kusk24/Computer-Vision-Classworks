# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

import cv2
import numpy as np

# File paths for the model and labels
weights_path = "/Users/kusk/Desktop/Computer Vision/Week4/frozen_inference_graph.pb"
config_path = "/Users/kusk/Desktop/Computer Vision/Week4/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
labels_path = "/Users/kusk/Desktop/Computer Vision/Week4/coco.names"
 
# Load class labels (COCO dataset)
with open(labels_path, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')
 
# Load the pre-trained model
net = cv2.dnn_DetectionModel(weights_path, config_path)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
 
# Read an image
img = cv2.imread("/Users/kusk/Desktop/Computer Vision/Week4/test_image.jpg")
 
# Detect objects
class_ids, confidences, boxes = net.detect(img, confThreshold=0.5)
 
# Draw results
for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
    label = f"{class_names[class_id - 1]}: {confidence:.2f}"
    cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
 
# Show output
cv2.imshow("Object Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()