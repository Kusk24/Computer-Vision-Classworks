import cv2
import numpy as np
import matplotlib.pyplot as plt


def align_images(img1, img2, feature_type='SIFT'):
    """
    Align img2 to img1 using feature matching and homography
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize feature detector
    if feature_type == 'SIFT':
        detector = cv2.SIFT_create()
    else:  # ORB
        detector = cv2.ORB_create()

    # Find keypoints and descriptors
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    # Match features
    if feature_type == 'SIFT':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:50]
    else:  # ORB
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:50]

    # Extract matched points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find homography
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # Warp img2 to align with img1
    h, w = img1.shape[:2]
    aligned_img = cv2.warpPerspective(img2, H, (w, h))

    return aligned_img, H, matches


# Example usage
if __name__ == "__main__":
    img1 = cv2.imread('left.jpg')
    img2 = cv2.imread('right.jpg')
    if img1 is None or img2 is None:
        print("Place left.jpg and right.jpg in this folder.")
    else:
        aligned, H, matches = align_images(img1, img2)
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        plt.title('Image 1 (reference)')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        plt.title('Image 2')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
        plt.title('Image 2 aligned to Image 1')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
