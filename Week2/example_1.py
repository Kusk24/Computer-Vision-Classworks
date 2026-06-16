import cv2
import numpy as np
import matplotlib.pyplot as plt
 
def translate_image(image, x, y):
    """
    Shift image by x pixels horizontally and y pixels vertically
    """
    # Get image dimensions
    height, width = image.shape[:2]

    # Define translation matrix: [[1, 0, tx], [0, 1, ty]]
    translation_matrix = np.float32([[1, 0, x], [0, 1, y]])

    # Apply affine transformation
    translated = cv2.warpAffine(image, translation_matrix, (width, height))
    return translated
 
# Load image
img = cv2.imread('sample.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib
 
# Translate by (50, 100)
translated_img = translate_image(img, 50, 100)
 
# Display
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img)
axes[0].set_title('Original')
axes[1].imshow(translated_img)
axes[1].set_title('Translated (50, 100)')
plt.show()