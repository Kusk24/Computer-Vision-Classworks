import cv2
import numpy as np
import matplotlib.pyplot as plt
 
def scale_image(image, scale_x, scale_y, interpolation=cv2.INTER_LINEAR):
    """
    Scale image using different interpolation methods
    """
    height, width = image.shape[:2]

    # Calculate new dimensions
    new_width = int(width * scale_x)
    new_height = int(height * scale_y)

    # Resize image
    scaled = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    return scaled
 
def scale_image_fixed_width(image, target_width):
    """
    Scale image to fixed width while maintaining aspect ratio
    """
    height, width = image.shape[:2]
    aspect_ratio = height / width
    new_height = int(target_width * aspect_ratio)

    scaled = cv2.resize(image, (target_width, new_height))
    return scaled
 
# Load image
img = cv2.imread('sample.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
# Different scaling methods
scaled_linear = scale_image(img, 0.5, 0.5, cv2.INTER_LINEAR)
scaled_cubic = scale_image(img, 0.5, 0.5, cv2.INTER_CUBIC)
scaled_nearest = scale_image(img, 0.5, 0.5, cv2.INTER_NEAREST)
scaled_fixed = scale_image_fixed_width(img, 300)
 
# Display
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(img)
axes[0, 0].set_title('Original')
axes[0, 1].imshow(scaled_linear)
axes[0, 1].set_title('Linear (0.5x)')
axes[0, 2].imshow(scaled_cubic)
axes[0, 2].set_title('Cubic (0.5x)')
axes[1, 0].imshow(scaled_nearest)
axes[1, 0].set_title('Nearest (0.5x)')
axes[1, 1].imshow(scaled_fixed)
axes[1, 1].set_title('Fixed width (300px)')
plt.tight_layout()
plt.show()