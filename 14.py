import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# Load grayscale image
image = cv2.imread(r"C:\Users\shalu\Desktop\codes\MVPR\grayscale.jpg", 0)

if image is None:
    print("Error loading image")
    exit()

# -----------------------------
# REGION GROWING
# -----------------------------
seed = (100, 100)          # Seed point (row, column)
threshold = 15

rows, cols = image.shape
visited = np.zeros((rows, cols), dtype=bool)
region = np.zeros((rows, cols), dtype=np.uint8)

seed_value = int(image[seed])
stack = [seed]

while stack:
    x, y = stack.pop()

    if x < 0 or x >= rows or y < 0 or y >= cols:
        continue

    if visited[x, y]:
        continue

    visited[x, y] = True

    if abs(int(image[x, y]) - seed_value) <= threshold:
        region[x, y] = 255

        # 8-connected neighbors
        stack.extend([
            (x-1, y), (x+1, y),
            (x, y-1), (x, y+1),
            (x-1, y-1), (x-1, y+1),
            (x+1, y-1), (x+1, y+1)
        ])

# Color the grown region in red
region_result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
region_result[region == 255] = [0, 0, 255]


# -----------------------------
# SPLIT-AND-MERGE SEGMENTATION
# -----------------------------

# Make image square with power-of-2 size
size = min(image.shape)
size = 2 ** int(np.floor(np.log2(size)))
img = image[:size, :size]

split_result = np.zeros((size, size, 3), dtype=np.uint8)

split_threshold = 15
min_size = 16

def split(img, x, y, block_size):

    block = img[x:x+block_size, y:y+block_size]

    if np.std(block) < split_threshold or block_size <= min_size:

        color = [
            random.randint(0,255),
            random.randint(0,255),
            random.randint(0,255)
        ]

        split_result[x:x+block_size,
                     y:y+block_size] = color
        return

    half = block_size // 2

    split(img, x, y, half)
    split(img, x, y+half, half)
    split(img, x+half, y, half)
    split(img, x+half, y+half, half)

split(img, 0, 0, size)


# -----------------------------
# DISPLAY RESULTS
# -----------------------------

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(image, cmap='gray')
plt.scatter(seed[1], seed[0], color='red', s=40)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(region_result, cv2.COLOR_BGR2RGB))
plt.title("Region Growing")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(split_result, cv2.COLOR_BGR2RGB))
plt.title("Split-and-Merge")
plt.axis('off')

plt.tight_layout()
plt.show()