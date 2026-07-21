import cv2
import numpy as np
import matplotlib.pyplot as plt


def stitch_images_opencv(images):
    """
    Use OpenCV's built-in stitcher
    """
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    status, stitched = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        return stitched
    else:
        print(f"Stitching failed with status: {status}")
        return None


# Example usage
if __name__ == "__main__":
    img1 = cv2.imread('left.jpg')
    img2 = cv2.imread('right.jpg')
    if img1 is None or img2 is None:
        print("Place left.jpg and right.jpg in this folder.")
    else:
        result = stitch_images_opencv([img1, img2])
        if result is not None:
            plt.figure(figsize=(15, 7))
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            plt.title('OpenCV Built-in Stitcher')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
