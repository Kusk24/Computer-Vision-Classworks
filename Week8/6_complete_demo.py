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


# Reuse functions from files 1 and 2 (single source of truth)
align_images = _load('1_basic_image_alignment.py', 'align_images')
stitch_images_manual = _load('2_simple_image_stitching.py', 'stitch_images_manual')


def demo_stitching():
    """
    Complete demo showing the stitching pipeline
    """
    # Load images (replace with your own images)
    img1 = cv2.imread('left.jpg')
    img2 = cv2.imread('right.jpg')

    # For demonstration with synthetic images (kept for reference):
    # img1 = np.zeros((300, 400, 3), dtype=np.uint8)
    # img2 = np.zeros((300, 400, 3), dtype=np.uint8)
    # cv2.rectangle(img1, (50, 50), (150, 150), (0, 255, 0), -1)
    # cv2.rectangle(img2, (50, 50), (150, 150), (255, 0, 0), -1)
    # cv2.circle(img1, (200, 150), 50, (255, 0, 0), -1)
    # cv2.circle(img2, (200, 150), 50, (0, 255, 0), -1)

    if img1 is None or img2 is None:
        print("Place left.jpg and right.jpg in this folder.")
        return None

    # Align images
    aligned, H, matches = align_images(img1, img2)

    # Stitch
    stitched = stitch_images_manual(img1, img2)

    # Display results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title('Image 1')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title('Image 2')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
    plt.title('Stitched')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return stitched


# Run demo
if __name__ == "__main__":
    result = demo_stitching()
