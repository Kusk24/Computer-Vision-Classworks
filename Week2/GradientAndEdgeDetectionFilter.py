import cv2
import numpy as np
import matplotlib.pyplot as plt

 

def edge_detection_comparison(image):
    """
    Compare different edge detection methods
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Sobel operator (gradient in x and y)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))

    # 2. Scharr operator (more accurate than Sobel)
    scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    scharr_combined = cv2.magnitude(scharr_x, scharr_y)
    scharr_combined = np.uint8(np.clip(scharr_combined, 0, 255))

    # 3. Laplacian (second derivative)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    laplacian = np.uint8(np.clip(np.abs(laplacian), 0, 255))

    # 4. Canny edge detector
    canny = cv2.Canny(gray, 100, 200)

    return sobel_combined, scharr_combined, laplacian, canny

 

img = cv2.imread('sample.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

 

sobel, scharr, laplacian, canny = edge_detection_comparison(img)

 

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes[0, 0].imshow(sobel, cmap='gray')
axes[0, 0].set_title('Sobel')
axes[0, 1].imshow(scharr, cmap='gray')
axes[0, 1].set_title('Scharr')
axes[1, 0].imshow(laplacian, cmap='gray')
axes[1, 0].set_title('Laplacian')
axes[1, 1].imshow(canny, cmap='gray')
axes[1, 1].set_title('Canny')
plt.tight_layout()
plt.show()

