import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread(r"C:\Users\user15\Desktop\FL\MVPR\lab cycle\image.jpg")

if image is None:
    print("Error loading image")
    exit()

# Convert BGR to RGB for display
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(10,4))

# Original Image
plt.subplot(1,2,1)
plt.imshow(rgb_image)
plt.title("Original Color Image")
plt.axis("off")

# Grayscale Image
plt.subplot(1,2,2)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis("off")

plt.show()

hist, bins = np.histogram(gray.flatten(), 256, [0,256])

plt.figure(figsize=(8,5))

plt.hist(gray.ravel(), bins=256, range=[0,256])

plt.title("Histogram of Grayscale Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.show()

mean_intensity = np.mean(gray)
min_intensity = np.min(gray)
max_intensity = np.max(gray)

print("\nANALYSIS OF HISTOGRAM")

print(f"Mean Intensity     : {mean_intensity:.2f}")
print(f"Minimum Intensity  : {min_intensity}")
print(f"Maximum Intensity  : {max_intensity}")

# Brightness Analysis
if mean_intensity < 85:
    print("Image is Dark")
elif mean_intensity > 170:
    print("Image is Bright")
else:
    print("Image has Moderate Brightness")

# Contrast Analysis
if (max_intensity - min_intensity) < 80:
    print("Image has Low Contrast")
else:
    print("Image has Good Contrast")

# Pixel Distribution Analysis
dark_pixels = np.sum(gray < 85)
mid_pixels = np.sum((gray >= 85) & (gray <= 170))
bright_pixels = np.sum(gray > 170)

total_pixels = gray.size

print("\nPixel Distribution:")

print(f"Dark Pixels      : {(dark_pixels/total_pixels)*100:.2f}%")
print(f"Mid-tone Pixels  : {(mid_pixels/total_pixels)*100:.2f}%")
print(f"Bright Pixels    : {(bright_pixels/total_pixels)*100:.2f}%")