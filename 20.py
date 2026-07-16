import cv2
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

# Load grayscale image
image = cv2.imread(r"C:\Users\shalu\Desktop\codes\MVPR\grayscale.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Could not load image.")
    exit()

# Define two Regions of Interest (ROIs)
h, w = image.shape

roi1 = image[0:h//2, 0:w//2]       # Top-left region
roi2 = image[h//2:h, w//2:w]       # Bottom-right region

# Function to compute GLCM features
def glcm_features(img):
    glcm = graycomatrix(img,
                        distances=[1],
                        angles=[0],
                        levels=256,
                        symmetric=True,
                        normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    return contrast, correlation, energy, homogeneity

# Compute features for each ROI
features1 = glcm_features(roi1)
features2 = glcm_features(roi2)

# Print Results
print("Region 1 (Top Left)")
print("Contrast     :", features1[0])
print("Correlation  :", features1[1])
print("Energy       :", features1[2])
print("Homogeneity  :", features1[3])

print("\nRegion 2 (Bottom Right)")
print("Contrast     :", features2[0])
print("Correlation  :", features2[1])
print("Energy       :", features2[2])
print("Homogeneity  :", features2[3])

# Display Image and ROIs
plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(roi1, cmap='gray')
plt.title("ROI 1")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(roi2, cmap='gray')
plt.title("ROI 2")
plt.axis("off")

plt.tight_layout()
plt.show()