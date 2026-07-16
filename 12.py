import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
image = cv2.imread(r"C:\Users\shalu\Desktop\codes\MVPR\grayscale.jpg", 0)

if image is None:
    print("Error loading image")
    exit()

# Convert to binary image
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Structuring Element (5x5 Square)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Opening = Erosion followed by Dilation
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Closing = Dilation followed by Erosion
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Display Results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(binary, cmap='gray')
plt.title("Original Noisy Binary Image")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(opening, cmap='gray')
plt.title("Opening (Erosion → Dilation)")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(closing, cmap='gray')
plt.title("Closing (Dilation → Erosion)")
plt.axis('off')

plt.tight_layout()
plt.show()