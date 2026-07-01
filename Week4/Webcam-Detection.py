import cv2

# Model and label paths
weights_path = "/Users/kusk/Desktop/Computer Vision/Week4/frozen_inference_graph.pb"
config_path = "/Users/kusk/Desktop/Computer Vision/Week4/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
labels_path = "/Users/kusk/Desktop/Computer Vision/Week4/coco.names"

with open(labels_path, 'rt') as f:
  class_names = f.read().rstrip('\n').split('\n')

# Load network
net = cv2.dnn_DetectionModel(weights_path, config_path)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
  ret, frame = cap.read()
  if not ret:
       break

  class_ids, confidences, boxes = net.detect(frame, confThreshold=0.5)

  for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
       cv2.rectangle(frame, box, (0, 255, 0), 2)
       label = f"{class_names[class_id - 1]}: {confidence:.2f}"
       cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

  cv2.imshow("Real-Time Detection", frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
       break

cap.release()
cv2.destroyAllWindows()
