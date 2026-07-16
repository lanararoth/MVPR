import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load binary image
image = cv2.imread(r"C:\Users\shalu\Desktop\codes\MVPR\grayscale.jpg", 0)

if image is None:
    print("Error loading image")
    exit()

# Convert to binary
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Connected Component Analysis
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

# Create a colored output image
colored_labels = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)

# Assign random colors to each object
colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
colors[0] = [0, 0, 0]   # Background is black

for i in range(num_labels):
    colored_labels[labels == i] = colors[i]

# Display images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(binary, cmap='gray')
plt.title("Binary Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(colored_labels, cv2.COLOR_BGR2RGB))
plt.title("Labeled Components")
plt.axis('off')

plt.tight_layout()
plt.show()

# Print object count and area
print("Total Number of Objects :", num_labels - 1)   # Excluding background

print("\nArea of Each Object:")
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    print(f"Object {i}: Area = {area} pixels")