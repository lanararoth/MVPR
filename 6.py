import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
image = cv2.imread(r"C:\Users\shalu\Desktop\codes\MVPR\grayscale.jpg")

if image is None:
    print("Error loading image")
    exit()

# Apply Laplacian filter
laplacian = cv2.Laplacian(image, cv2.CV_64F)

# Convert to absolute values for display
laplacian_abs = cv2.convertScaleAbs(laplacian)

# Edge enhancement
enhanced = cv2.addWeighted(image, 1.0, laplacian_abs, 1.0, 0)

# Display results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(laplacian_abs, cmap='gray')
plt.title("Laplacian Edges")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(enhanced, cmap='gray')
plt.title("Enhanced Image")
plt.axis('off')

plt.tight_layout()
plt.show()