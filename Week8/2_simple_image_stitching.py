import cv2
import numpy as np
import matplotlib.pyplot as plt


def stitch_images_manual(img1, img2):
    """
    Manually stitch two images using homography
    """
    # Detect keypoints and compute homography
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Match features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Get matching points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # Calculate stitching dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Get corners of img2
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    corners2_transformed = cv2.perspectiveTransform(corners2, H)

    # Combine corners to get final image size
    all_corners = np.concatenate([corners2_transformed,
                                 np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)])

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Translation to shift to positive coordinates
    translation = np.float32([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    # Warp img2 with translation
    H_translated = translation @ H
    warped_img2 = cv2.warpPerspective(img2, H_translated, (x_max - x_min, y_max - y_min))

    # Create output image
    stitched = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
    stitched[-y_min:h1 - y_min, -x_min:w1 - x_min] = img1

    # Add warped image (simple overlay, can be improved with blending)
    mask = (warped_img2 > 0)
    stitched[mask] = warped_img2[mask]

    return stitched


# Example usage
if __name__ == "__main__":
    img1 = cv2.imread('left.jpg')
    img2 = cv2.imread('right.jpg')
    if img1 is None or img2 is None:
        print("Place left.jpg and right.jpg in this folder.")
    else:
        stitched = stitch_images_manual(img1, img2)
        plt.figure(figsize=(15, 7))
        plt.imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
        plt.title('Manual Stitch (SIFT + RANSAC)')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
