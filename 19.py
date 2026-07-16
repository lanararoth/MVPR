import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# Load grayscale image
image = cv2.imread(r"C:\Users\shalu\Desktop\codes\MVPR\image.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Could not load image.")
    exit()

# LBP Parameters
radius = 1
n_points = 8 * radius

# Compute LBP
lbp = local_binary_pattern(image, n_points, radius, method='uniform')

# Compute Histogram
n_bins = int(lbp.max() + 1)
hist, _ = np.histogram(lbp.ravel(),
                       bins=n_bins,
                       range=(0, n_bins))

# Normalize Histogram
hist = hist.astype("float")
hist /= (hist.sum() + 1e-7)

# Display Images
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(lbp, cmap='gray')
plt.title("LBP Image")
plt.axis("off")

plt.subplot(1,3,3)
plt.bar(range(len(hist)), hist)
plt.title("LBP Histogram")
plt.xlabel("LBP Value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()