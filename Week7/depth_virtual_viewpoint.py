import cv2
import numpy as np

def render_with_depth(color_img, depth_img, shift_amount, baseline):
    """
    Renders a new viewpoint from a color image and its depth map.
    Uses a simple 3D warping approximation.
    """
    h, w = color_img.shape[:2]
    # Normalize the depth image to a range between 0 and 1
    if depth_img.max() > 1:
        depth_norm = depth_img.astype(np.float32) / 255.0
    else:
        depth_norm = depth_img.astype(np.float32)

    # Create a 3D grid of pixel coordinates
    y, x = np.mgrid[0:h, 0:w]
    # Convert pixel coordinates to be centered at the image center
    x = x - w / 2
    y = y - h / 2

    # Focal length and baseline to control disparity
    # Disparity is inversely proportional to depth
    focal_length = 500
    # To avoid division by zero in a real scenario, add a small epsilon
    # Here we use a trick: disparity = 1 / (depth + 0.1)
    depth_epsilon = 0.1
    # Calculate disparity. For objects with high depth (white), disparity is small.
    disparity = baseline * focal_length / (depth_norm * 100 + depth_epsilon)

    # Displace pixels based on depth to synthesize a new viewpoint
    # For a shift in viewpoint, pixels with small depth (close) move more
    map_x = (x + disparity * shift_amount + w / 2).astype(np.float32)
    map_y = (y + h / 2).astype(np.float32)

    # Remap the color image using the calculated maps
    # Warp mode: cv2.INTER_LINEAR for smooth interpolation
    new_view = cv2.remap(color_img, map_x, map_y, cv2.INTER_LINEAR)
    return new_view

# Create synthetic color and depth images
# A 300x300 black background
color_img = np.zeros((300, 300, 3), dtype=np.uint8)
# A 300x300 depth map (0-255, where 0 is close, 255 is far)
depth_img = np.zeros((300, 300), dtype=np.uint8)

# Draw an object (a red square) and define its depth
cv2.rectangle(color_img, (100, 100), (200, 200), (0, 0, 255), -1)
# Assign a close depth (dark) to the red square
cv2.rectangle(depth_img, (100, 100), (200, 200), 50, -1)
# Assign a far depth (light) to the background
depth_img = cv2.bitwise_not(depth_img)  # Inverts depth so black is close, white is far

# Simulate a baseline (distance between cameras) for a stereo system
baseline = 2.0

# Render a new viewpoint by shifting right (positive shift amount)
new_view = render_with_depth(color_img, depth_img, shift_amount=0.5, baseline=baseline)

cv2.imshow('Original View', color_img)
cv2.imshow('Depth Map', depth_img)
cv2.imshow('Rendered New Viewpoint', new_view)
print("Rendering a new viewpoint using depth information.")
print("The red square (close object) moves more than the background.")
print("Press any key to close the windows.")
cv2.waitKey(0)
cv2.destroyAllWindows()
