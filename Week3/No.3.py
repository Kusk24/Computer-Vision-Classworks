# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load an image
image = cv2.imread('Week3/sample.jpg')  # Replace with your image path
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Detect edges
edges = cv2.Canny(gray, 50, 150)

# 3. Find edge points (non-zero pixels)
edge_points = np.column_stack(np.where(edges > 0))

# 4. Skip if no edges found
if len(edge_points) > 0:
    # 5. Fit a line to edge points using least squares
    x = edge_points[:, 1]  # Column indices
    y = edge_points[:, 0]  # Row indices

    # Least squares fitting
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    # 6. Display results
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(edges, cmap='gray')
    plt.title('Edges')

    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    # Plot the fitted line
    line_x = np.array([0, edges.shape[1]])
    line_y = m * line_x + c
    plt.plot(line_x, line_y, 'r-', linewidth=2)
    plt.title('Fitted Line on Edges')

    plt.tight_layout()
    plt.show()
else:
    print("No edges detected in the image.")
