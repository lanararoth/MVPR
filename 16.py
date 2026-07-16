import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load binary image
image = cv2.imread(r"C:\Users\shalu\Desktop\codes\MVPR\image.jpg")

if image is None:
    print("Error loading image")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold to binary
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(binary,
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

# Copy images
bounding_img = image.copy()
rectangle_img = image.copy()

count = 0

for cnt in contours:

    # Ignore very small objects (noise)
    if cv2.contourArea(cnt) < 100:
        continue

    count += 1

    # Bounding Box
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(bounding_img,
                  (x, y),
                  (x + w, y + h),
                  (0, 255, 0),
                  2)

    # Minimum Enclosing Rectangle
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    cv2.drawContours(rectangle_img,
                     [box],
                     0,
                     (255, 0, 0),
                     2)

# Display Results
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(binary, cmap='gray')
plt.title("Binary Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(bounding_img, cv2.COLOR_BGR2RGB))
plt.title("Bounding Boxes")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(rectangle_img, cv2.COLOR_BGR2RGB))
plt.title("Minimum Enclosing Rectangles")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Total Number of Objects:", count)