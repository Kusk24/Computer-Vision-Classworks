import cv2
import numpy as np
import matplotlib.pyplot as plt
 
def rotate_image(image, angle, center=None, scale=1.0):
    """
    Rotate image by given angle around center point
    """
    height, width = image.shape[:2]

    # If center not specified, use image center
    if center is None:
        center = (width // 2, height // 2)

    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # Apply rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated
 
def rotate_without_cropping(image, angle):
    """
    Rotate image without cropping - expands canvas to fit rotated image
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new dimensions to avoid cropping
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])

    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    # Adjust rotation matrix to account for translation
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    # Apply rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
    return rotated
 
# Load image
img = cv2.imread('sample.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
# Rotate 45 degrees
rotated_img = rotate_image(img, 45)
rotated_no_crop = rotate_without_cropping(img, 45)
 
# Display
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img)
axes[0].set_title('Original')
axes[1].imshow(rotated_img)
axes[1].set_title('Rotated 45° (cropped)')
axes[2].imshow(rotated_no_crop)
axes[2].set_title('Rotated 45° (no crop)')
plt.show()