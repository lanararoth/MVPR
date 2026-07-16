import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
image = cv2.imread(r"C:\Users\shalu\Desktop\codes\MVPR\grayscale.jpg")

if image is None:
    print("Error loading image")
    exit()

# ---------- Sobel Operator ----------

sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

sobel = np.sqrt(sobel_x**2 + sobel_y**2)
sobel = cv2.convertScaleAbs(sobel)

# ---------- Prewitt Operator ----------

prewitt_x_kernel = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]])

prewitt_y_kernel = np.array([[-1, -1, -1],
                             [ 0,  0,  0],
                             [ 1,  1,  1]])

prewitt_x = cv2.filter2D(image, -1, prewitt_x_kernel)
prewitt_y = cv2.filter2D(image, -1, prewitt_y_kernel)

prewitt = np.sqrt(prewitt_x.astype(np.float64)**2 +
                  prewitt_y.astype(np.float64)**2)
prewitt = cv2.convertScaleAbs(prewitt)

# ---------- Display Results ----------

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(sobel, cmap='gray')
plt.title("Sobel Edge Detection")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(prewitt, cmap='gray')
plt.title("Prewitt Edge Detection")
plt.axis('off')

plt.tight_layout()
plt.show()