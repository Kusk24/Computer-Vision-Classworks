# Name - Win Yu Maung
# ID - 6612054
# Sec - 541 
 
import cv2
import numpy as np
 
# Load pre-trained model (e.g., from OpenCV's model zoo)
# For this example, we'll use a placeholder. In practice, you would download a model like ResNet.
net = cv2.dnn.readNetFromCaffe("/Users/kusk/Desktop/Computer Vision/Week4/deploy.prototxt", "/Users/kusk/Desktop/Computer Vision/Week4/model.caffemodel")  # Example for Caffe
 
# Alternatively, for TensorFlow:
# net = cv2.dnn.readNetFromTensorflow("frozen_inference_graph.pb", "graph.pbtxt")
 
# Load and preprocess image
image = cv2.imread("/Users/kusk/Desktop/Computer Vision/Week4/test_image.jpg")
blob = cv2.dnn.blobFromImage(image, scalefactor=1.0/127.5, size=(224, 224), mean=(127.5, 127.5, 127.5), swapRB=True)
net.setInput(blob)
 
# Forward pass
output = net.forward()
output = output.flatten()

# Get top-5 predictions
indices = np.argsort(output)[::-1][:5]
for i, idx in enumerate(indices):
   print(f"{i+1}: Class {idx}, Score: {output[idx]:.3f}")

# Show image with top prediction
top_idx = indices[0]
label = f"Class {top_idx}: {output[top_idx]:.2f}"
cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow("Image Classification", image)
cv2.waitKey(0)
cv2.destroyAllWindows()