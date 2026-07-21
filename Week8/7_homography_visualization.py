import os
import importlib.util
import cv2
import numpy as np
import matplotlib.pyplot as plt


def _load(filename, attr):
    """Load a function from a sibling file whose name starts with a digit."""
    path = os.path.join(os.path.dirname(__file__), filename)
    spec = importlib.util.spec_from_file_location(filename.replace('.py', ''), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attr)


# Reuse align_images from file 1 (single source of truth)
align_images = _load('1_basic_image_alignment.py', 'align_images')


def visualize_homography(img1, img2):
    """
    Visualize the homography transformation
    """
    # Get homography
    _, H, _ = align_images(img1, img2)

    # Get image corners
    h, w = img2.shape[:2]
    corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, H)

    # Draw on image
    img1_copy = img1.copy()
    img1_copy = cv2.polylines(img1_copy, [np.int32(transformed_corners)], True, (0, 255, 0), 3)

    # Combine images for display
    combined = np.hstack([img1_copy, img2])

    return combined


# Example usage
if __name__ == "__main__":
    img1 = cv2.imread('left.jpg')
    img2 = cv2.imread('right.jpg')
    if img1 is None or img2 is None:
        print("Place left.jpg and right.jpg in this folder.")
    else:
        combined = visualize_homography(img1, img2)
        plt.figure(figsize=(15, 7))
        plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        plt.title('Homography Visualization (green = where img2 maps into img1)')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
