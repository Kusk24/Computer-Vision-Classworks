import cv2
import numpy as np
import matplotlib.pyplot as plt
 
def compare_blur_filters(image):
    """
    Apply and compare different blurring techniques
    """
    # 1. Average blur (box filter)
    avg_blur = cv2.blur(image, (15, 15))
    # 2. Gaussian blur
    gaussian_blur = cv2.GaussianBlur(image, (15, 15), 0)
    # 3. Median blur (good for salt & pepper noise)
    median_blur = cv2.medianBlur(image, 15)
    # 4. Bilateral filter (preserves edges)
    bilateral = cv2.bilateralFilter(image, 15, 75, 75)
    return avg_blur, gaussian_blur, median_blur, bilateral
 
# Load and prepare image
img = cv2.imread('sample.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
# Add noise for demonstration
def add_noise(image):
    noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
    noisy = cv2.add(image, noise)
    return noisy
 
noisy_img = add_noise(img_rgb)
 
# Apply filters
avg_blur, gaussian_blur, median_blur, bilateral = compare_blur_filters(noisy_img)
 
# Display
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(noisy_img)
axes[0, 0].set_title('Noisy Image')
axes[0, 1].imshow(avg_blur)
axes[0, 1].set_title('Average Blur')
axes[0, 2].imshow(gaussian_blur)
axes[0, 2].set_title('Gaussian Blur')
axes[1, 0].imshow(median_blur)
axes[1, 0].set_title('Median Blur')
axes[1, 1].imshow(bilateral)
axes[1, 1].set_title('Bilateral Filter')
axes[1, 2].axis('off')
plt.tight_layout()
plt.show()