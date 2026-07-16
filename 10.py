import cv2
import matplotlib.pyplot as plt

# Load image in grayscale
image = cv2.imread(r"C:\Users\shalu\Desktop\codes\MVPR\grayscale.jpg", 0)

if image is None:
    print("Error loading image")
    exit()

# Global Thresholding
threshold_value = 127
_, global_thresh = cv2.threshold(
    image, threshold_value, 255, cv2.THRESH_BINARY
)

# Otsu's Automatic Thresholding
otsu_threshold, otsu_thresh = cv2.threshold(
    image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Display Results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Grayscale Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(global_thresh, cmap='gray')
plt.title(f"Global Threshold\n(T = {threshold_value})")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(otsu_thresh, cmap='gray')
plt.title(f"Otsu Threshold\n(T = {otsu_threshold:.0f})")
plt.axis('off')

plt.tight_layout()
plt.show()

# Print threshold values
print("Global Threshold Value :", threshold_value)
print("Otsu Threshold Value   :", otsu_threshold)