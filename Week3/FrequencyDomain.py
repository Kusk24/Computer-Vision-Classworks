# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

import cv2
import numpy as np
import matplotlib.pyplot as plt

def frequency_filter(image, filter_type='lowpass', cutoff=30):
    """
    Apply filtering in frequency domain
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform FFT
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # Create filter
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Create meshgrid for distance calculation
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((y - crow)**2 + (x - ccol)**2)

    if filter_type == 'lowpass':
        # Ideal low-pass filter
        mask = distance <= cutoff
    elif filter_type == 'highpass':
        # Ideal high-pass filter
        mask = distance > cutoff
    elif filter_type == 'gaussian_lowpass':
        # Gaussian low-pass
        mask = np.exp(-(distance**2) / (2 * (cutoff**2)))
    elif filter_type == 'gaussian_highpass':
        # Gaussian high-pass
        mask = 1 - np.exp(-(distance**2) / (2 * (cutoff**2)))
    else:
        mask = np.ones((rows, cols))

    # Apply filter
    fshift_filtered = fshift * mask

    # Inverse FFT
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_filtered = np.fft.ifft2(f_ishift)
    img_filtered = np.abs(img_filtered)
    img_filtered = np.uint8(np.clip(img_filtered, 0, 255))

    return image, img_filtered, np.log(1 + np.abs(fshift))

img = cv2.imread('Week3/sample.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply different frequency filters
original, lowpass, fft_mag = frequency_filter(img, 'lowpass', 30)
_, highpass, _ = frequency_filter(img, 'highpass', 30)
_, gauss_low, _ = frequency_filter(img, 'gaussian_lowpass', 30)
_, gauss_high, _ = frequency_filter(img, 'gaussian_highpass', 30)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(original, cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 1].imshow(lowpass, cmap='gray')
axes[0, 1].set_title('Ideal Low-Pass')
axes[0, 2].imshow(highpass, cmap='gray')
axes[0, 2].set_title('Ideal High-Pass')
axes[1, 0].imshow(gauss_low, cmap='gray')
axes[1, 0].set_title('Gaussian Low-Pass')
axes[1, 1].imshow(gauss_high, cmap='gray')
axes[1, 1].set_title('Gaussian High-Pass')
axes[1, 2].imshow(fft_mag, cmap='gray')
axes[1, 2].set_title('FFT Magnitude')
plt.tight_layout()
plt.show()
