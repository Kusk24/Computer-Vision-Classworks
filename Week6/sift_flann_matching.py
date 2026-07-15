import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 1. Load images (provide your own paths)
img1 = cv.imread('box.png', cv.IMREAD_GRAYSCALE)          # queryImage (object to find)
img2 = cv.imread('box_in_scene.png', cv.IMREAD_GRAYSCALE) # trainImage (scene to search)

if img1 is None or img2 is None:
    print("Error: Could not load images.")
    exit()

# 2. Initialize SIFT detector
sift = cv.SIFT_create()

# 3. Find keypoints and descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 4. Set up FLANN parameters
# For SIFT (floating-point descriptors), we use a KD-Tree
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)   # Higher = more accurate but slower

flann = cv.FlannBasedMatcher(index_params, search_params)

# 5. Perform k-NN matching (k=2 to get best two matches for ratio test)
matches = flann.knnMatch(des1, des2, k=2)

# 6. Apply Lowe's ratio test (from Szeliski Ch. 7.1.3)
# This filters good matches by comparing the best match distance to the second best.
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:  # threshold can be tuned
        good_matches.append(m)

# 7. Visualize the results
img_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(15, 7))
plt.imshow(img_matches)
plt.title(f'FLANN + SIFT Matches (Lowe\'s Ratio Test, {len(good_matches)} matches)')
plt.axis('off')
plt.show()
