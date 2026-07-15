import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 1. Load images
img1 = cv.imread('box.png', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('box_in_scene.png', cv.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Error: Could not load images.")
    exit()

# 2. Initialize SIFT detector
sift = cv.SIFT_create()

# 3. Find keypoints and descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 4. Create a Brute-Force Matcher (L2 norm for SIFT)
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)  # crossCheck=False to use ratio test

# 5. Perform k-NN matching (k=2)
matches = bf.knnMatch(des1, des2, k=2)

# 6. Apply Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:  # Slightly different threshold
        good_matches.append(m)

# 7. Draw and display matches
img_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(15, 7))
plt.imshow(img_matches)
plt.title(f'BF + SIFT Matches ({len(good_matches)} matches)')
plt.axis('off')
plt.show()
