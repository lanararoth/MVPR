import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
image = cv2.imread(r"C:\Users\shalu\Desktop\codes\MVPR\grayscale.jpg", 0)

if image is None:
    print("Error loading image")
    exit()

# Compute 2D Fourier Transform
f = np.fft.fft2(image)

# Shift zero-frequency component to center
fshift = np.fft.fftshift(f)

# Compute Magnitude Spectrum
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

# Display sample Fourier coefficients
print("\nFirst 10x10 Fourier Coefficients:")
print(f[:10, :10])

# Display sample Magnitude values
print("\nFirst 10x10 Magnitude Values:")
print(np.abs(fshift[:10, :10]))

# Display results
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Grayscale Image")
plt.axis('off')

# Magnitude Spectrum
plt.subplot(1, 2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title("Magnitude Spectrum")
plt.axis('off')

plt.tight_layout()
plt.show()

# Fourier computation details
print("\nImage Shape:", image.shape)
print("Fourier Transform Shape:", f.shape)
print("Maximum Magnitude:", np.max(np.abs(fshift)))
print("Minimum Magnitude:", np.min(np.abs(fshift)))