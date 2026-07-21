import cv2
import numpy as np
import matplotlib.pyplot as plt


def align_with_template(img1, img2):
    """
    Simple alignment using template matching for small translations
    Useful for images with small translational differences
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Use center region of img2 as template
    h, w = gray2.shape
    template = gray2[h//4:3*h//4, w//4:3*w//4]

    # Template matching
    result = cv2.matchTemplate(gray1, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Get translation
    tx = max_loc[0] - w//4
    ty = max_loc[1] - h//4

    # Apply translation to img2
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    aligned = cv2.warpAffine(img2, M, (w, h))

    return aligned


# Example usage
if __name__ == "__main__":
    img1 = cv2.imread('left.jpg')   # reference
    img2 = cv2.imread('right.jpg')  # to_align
    if img1 is None or img2 is None:
        print("Place left.jpg and right.jpg in this folder.")
    else:
        aligned = align_with_template(img1, img2)
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        plt.title('Reference')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        plt.title('To align')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
        plt.title('Aligned (template matching)')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
