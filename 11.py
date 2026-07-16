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

# Structuring Elements
square3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
square5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

circle3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
circle5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Dilation
dilate_square = cv2.dilate(binary, square5, iterations=1)
dilate_circle = cv2.dilate(binary, circle5, iterations=1)

# Erosion
erode_square = cv2.erode(binary, square5, iterations=1)
erode_circle = cv2.erode(binary, circle5, iterations=1)

# Display Results
plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.imshow(binary, cmap='gray')
plt.title("Original Binary")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(dilate_square, cmap='gray')
plt.title("Dilation (Square 5×5)")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(dilate_circle, cmap='gray')
plt.title("Dilation (Circle 5×5)")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(erode_square, cmap='gray')
plt.title("Erosion (Square 5×5)")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(erode_circle, cmap='gray')
plt.title("Erosion (Circle 5×5)")
plt.axis('off')

plt.tight_layout()
plt.show()