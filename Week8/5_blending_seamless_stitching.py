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


# Reuse the manual stitcher from file 2 (single source of truth)
stitch_images_manual = _load('2_simple_image_stitching.py', 'stitch_images_manual')


def blend_images(img1, img2, mask_width=30):
    """
    Blend two images with a linear gradient in the overlap region
    """
    h, w = img1.shape[:2]

    # Create blending mask
    mask = np.zeros((h, w), dtype=np.float32)

    # Assume vertical stitching with overlap on the right
    overlap_start = w - mask_width
    for i in range(w):
        if i < overlap_start:
            mask[:, i] = 1.0
        else:
            alpha = 1.0 - (i - overlap_start) / mask_width
            mask[:, i] = alpha

    # Expand mask to 3 channels
    mask_3ch = np.stack([mask, mask, mask], axis=2)

    # Blend images
    blended = (img1 * mask_3ch + img2 * (1 - mask_3ch)).astype(np.uint8)

    return blended


def stitch_with_blending(img1, img2):
    """
    Stitch two images with blending
    """
    # First, get the stitched image (using manual stitching function)
    stitched = stitch_images_manual(img1, img2)

    # Alternative: Use OpenCV's built-in seamless cloning for better results
    # This is more advanced and requires careful parameter tuning

    return stitched


# Example usage
if __name__ == "__main__":
    img1 = cv2.imread('left.jpg')
    img2 = cv2.imread('right.jpg')
    if img1 is None or img2 is None:
        print("Place left.jpg and right.jpg in this folder.")
    else:
        # blend_images needs two same-size images; resize img2 to match img1
        img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        blended = blend_images(img1, img2_resized, mask_width=100)
        plt.figure(figsize=(10, 7))
        plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        plt.title('Linear Gradient Blend (overlap region)')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
