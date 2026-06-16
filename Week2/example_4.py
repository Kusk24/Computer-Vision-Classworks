import cv2
import numpy as np
import matplotlib.pyplot as plt
 
def affine_transform_image(image, src_points, dst_points):
    """
    Apply affine transformation using 3 pairs of corresponding points
    """

    # Get affine transformation matrix
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)

    # Apply transformation
    height, width = image.shape[:2]
    transformed = cv2.warpAffine(image, affine_matrix, (width, height))

    return transformed
 
def skew_image(image, skew_factor=0.3):
    """
    Apply skew transformation (shear) to image
    """

    height, width = image.shape[:2]

    # Define source points (corners of image)
    src_points = np.float32([
        [0, 0],
        [width-1, 0],
        [0, height-1]
    ])

    # Define destination points with skew
    dst_points = np.float32([
        [0, 0], 
        [width-1, 0],
        [skew_factor * height, height-1]
    ])

    return affine_transform_image(image, src_points, dst_points)
 
def mirror_image(image, mirror_axis='vertical'):
    """
    Mirror (flip) image along specified axis
    """

    if mirror_axis == 'vertical':
        flipped = cv2.flip(image, 1)  # 1 = horizontal flip

    elif mirror_axis == 'horizontal':
        flipped = cv2.flip(image, 0)  # 0 = vertical flip

    elif mirror_axis == 'both':
        flipped = cv2.flip(image, -1)  # -1 = both axes

    else:
        flipped = image

    return flipped
 
# Load image
img = cv2.imread('sample.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
# Apply different affine transforms
skewed_img = skew_image(img, 0.4)
mirrored_v = mirror_image(img, 'vertical')
mirrored_both = mirror_image(img, 'both')
 
# Custom affine transform (warping)
height, width = img.shape[:2]

src_pts = np.float32([
    [0, 0],
    [width-1, 0],
    [0, height-1]
])

dst_pts = np.float32([
    [width*0.1, height*0.1],
    [width*0.9, height*0.1],
    [width*0.3, height*0.8]
])

custom_affine = affine_transform_image(img, src_pts, dst_pts)
 
# Display
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

axes[0, 0].imshow(skewed_img)
axes[0, 0].set_title('Skewed')

axes[0, 1].imshow(mirrored_v)
axes[0, 1].set_title('Mirrored (Vertical)')

axes[1, 0].imshow(mirrored_both)
axes[1, 0].set_title('Mirrored (Both)')

axes[1, 1].imshow(custom_affine)
axes[1, 1].set_title('Custom Affine')

plt.tight_layout()
plt.show()