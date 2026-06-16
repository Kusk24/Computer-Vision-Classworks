import cv2
import numpy as np
import matplotlib.pyplot as plt
 
def perspective_transform(image, src_points, dst_points):
    """
    Apply perspective transformation using 4 pairs of corresponding points
    """

    # Get perspective transformation matrix
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply transformation
    height, width = image.shape[:2]
    transformed = cv2.warpPerspective(image, perspective_matrix, (width, height))

    return transformed
 
def correct_perspective(image):
    """
    Correct perspective distortion in an image
    (e.g., for document scanning)
    """

    height, width = image.shape[:2]

    # Define source points
    src_points = np.float32([
        [width*0.2, height*0.2],
        [width*0.8, height*0.1],
        [width*0.1, height*0.9],
        [width*0.9, height*0.8]
    ])

    # Define destination points (rectangle)
    dst_points = np.float32([
        [0, 0],
        [width, 0],
        [0, height],
        [width, height]
    ])

    return perspective_transform(image, src_points, dst_points)
 
def zoom_region(image, region, zoom_factor=2.0):
    """
    Zoom into a specific region of the image
    using perspective transform
    """

    height, width = image.shape[:2]

    # Define source points (region to zoom)
    x, y, w, h = region

    src_points = np.float32([
        [x, y],
        [x+w, y],
        [x, y+h],
        [x+w, y+h]
    ])

    # Define destination points (full size)
    dst_points = np.float32([
        [0, 0],
        [width, 0],
        [0, height],
        [width, height]
    ])

    return perspective_transform(image, src_points, dst_points)
 
# Load image
img = cv2.imread('sample.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

height, width = img.shape[:2]
 
# Apply perspective transforms
perspective_corrected = correct_perspective(img)
 
# Zoom into center region
center_region = (
    width//4,
    height//4,
    width//2,
    height//2
)

zoomed_img = zoom_region(
    img,
    center_region,
    zoom_factor=2.0
)
 
# 3D perspective effect
src_pts_3d = np.float32([
    [0, 0],
    [width-1, 0],
    [0, height-1],
    [width-1, height-1]
])

dst_pts_3d = np.float32([
    [width*0.2, height*0.1],
    [width*0.8, height*0.2],
    [width*0.1, height*0.9],
    [width*0.9, height*0.8]
])

perspective_3d = perspective_transform(
    img,
    src_pts_3d,
    dst_pts_3d
)
 
# Display
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

axes[0, 0].imshow(perspective_corrected)
axes[0, 0].set_title('Perspective Corrected')

axes[0, 1].imshow(zoomed_img)
axes[0, 1].set_title('Zoomed Region (2x)')

axes[1, 0].imshow(perspective_3d)
axes[1, 0].set_title('3D Perspective Effect')

axes[1, 1].axis('off')

plt.tight_layout()
plt.show()