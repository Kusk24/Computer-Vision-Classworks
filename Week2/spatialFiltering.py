import cv2
import numpy as np
import matplotlib.pyplot as plt

 

def apply_custom_kernel(image, kernel):
    """
    Apply custom 2D convolution kernel to image
    """
    return cv2.filter2D(image, -1, kernel)

 

# Load image
img = cv2.imread('sample.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

 

# Define custom kernels
# 1. Identity kernel
identity_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

 

# 2. Edge detection (Sobel-like)
edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

 

# 3. Sharpen
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

 

# 4. Emboss
emboss_kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])

 

# Apply kernels
identity_img = apply_custom_kernel(img_rgb, identity_kernel)
edge_img = apply_custom_kernel(img_rgb, edge_kernel)
sharpen_img = apply_custom_kernel(img_rgb, sharpen_kernel)
emboss_img = apply_custom_kernel(img_rgb, emboss_kernel)

 

# Display results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title('Original')
axes[0, 1].imshow(identity_img)
axes[0, 1].set_title('Identity')
axes[0, 2].imshow(edge_img)
axes[0, 2].set_title('Edge Detection')
axes[1, 0].imshow(sharpen_img)
axes[1, 0].set_title('Sharpen')
axes[1, 1].imshow(emboss_img)
axes[1, 1].set_title('Emboss')
axes[1, 2].axis('off')
plt.tight_layout()
plt.show()