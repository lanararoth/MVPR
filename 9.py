import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Load image in grayscale
image = cv2.imread(r"C:\Users\shalu\Desktop\codes\MVPR\image.jpg", 0)

if image is None:
    print("Error loading image")
    exit()

# -----------------------------
# Spatial Domain Smoothing
# -----------------------------
start = time.time()
spatial_smooth = cv2.GaussianBlur(image, (7,7), 0)
spatial_smooth_time = time.time() - start

# -----------------------------
# Frequency Domain Smoothing
# Low-Pass Filter
# -----------------------------
start = time.time()

f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)

rows, cols = image.shape
crow, ccol = rows//2, cols//2

mask = np.zeros((rows, cols), np.uint8)
radius = 50

for i in range(rows):
    for j in range(cols):
        if (i-crow)**2 + (j-ccol)**2 <= radius**2:
            mask[i, j] = 1

low_pass = fshift * mask

f_ishift = np.fft.ifftshift(low_pass)
freq_smooth = np.fft.ifft2(f_ishift)
freq_smooth = np.abs(freq_smooth)

freq_smooth_time = time.time() - start

# -----------------------------
# Spatial Domain Sharpening
# Unsharp Masking
# -----------------------------
start = time.time()

blurred = cv2.GaussianBlur(image, (7,7), 0)
spatial_sharp = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

spatial_sharp_time = time.time() - start

# -----------------------------
# Frequency Domain Sharpening
# High-Pass Filter
# -----------------------------
start = time.time()

mask_hp = np.ones((rows, cols), np.uint8)

for i in range(rows):
    for j in range(cols):
        if (i-crow)**2 + (j-ccol)**2 <= radius**2:
            mask_hp[i, j] = 0

high_pass = fshift * mask_hp

f_ishift_hp = np.fft.ifftshift(high_pass)
freq_sharp = np.fft.ifft2(f_ishift_hp)
freq_sharp = np.abs(freq_sharp)

freq_sharp_time = time.time() - start

# -----------------------------
# Display Results
# -----------------------------
plt.figure(figsize=(15,10))

plt.subplot(2,3,1)
plt.imshow(image, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(spatial_smooth, cmap='gray')
plt.title("Spatial Smoothing")
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(freq_smooth, cmap='gray')
plt.title("Frequency Smoothing")
plt.axis('off')

plt.subplot(2,3,5)
plt.imshow(spatial_sharp, cmap='gray')
plt.title("Spatial Sharpening")
plt.axis('off')

plt.subplot(2,3,6)
plt.imshow(freq_sharp, cmap='gray')
plt.title("Frequency Sharpening")
plt.axis('off')

plt.tight_layout()
plt.show()

# -----------------------------
# Computation Time
# -----------------------------
print("\nCOMPUTATION TIME")
print("-"*40)
print(f"Spatial Smoothing   : {spatial_smooth_time:.6f} sec")
print(f"Frequency Smoothing : {freq_smooth_time:.6f} sec")
print(f"Spatial Sharpening  : {spatial_sharp_time:.6f} sec")
print(f"Frequency Sharpening: {freq_sharp_time:.6f} sec")