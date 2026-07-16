import cv2
import numpy as np
import matplotlib.pyplot as plt

# LOAD IMAGE IN GRAYSCALE

image = cv2.imread(r"C:\Users\shalu\Desktop\codes\MVPR\grayscale.jpg")

if image is None:
    print("Error loading image")
    exit()

# APPLY GAUSSIAN BLUR


blurred = cv2.GaussianBlur(image, (5, 5), 0)

# CREATE UNSHARP MASK


mask = cv2.subtract(image, blurred)

# DIFFERENT MASK WEIGHTS

weights = [0.5, 1.0, 1.5, 2.0]

# Store sharpened images
sharpened_images = []

for w in weights:
    sharpened = cv2.addWeighted(image, 1.0, mask, w, 0)
    sharpened_images.append(sharpened)

# DISPLAY RESULTS

plt.figure(figsize=(18, 5))

# Original Image
plt.subplot(1, 5, 1)
plt.imshow(image, cmap='gray')
plt.title("Original")
plt.axis('off')

# Sharpened Images
for i, (w, img) in enumerate(zip(weights, sharpened_images)):
    plt.subplot(1, 5, i + 2)
    plt.imshow(img, cmap='gray')
    plt.title(f"Weight = {w}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# IMAGE ANALYSIS

def edge_strength(img):
    """
    Calculate edge strength using Laplacian Variance.
    Higher value = Sharper image.
    """
    return cv2.Laplacian(img, cv2.CV_64F).var()

print("\n===== UNSHARP MASKING ANALYSIS =====\n")

original_strength = edge_strength(image)

print(f"Original Image")
print(f"- Edge Strength: {original_strength:.2f}\n")

for w, img in zip(weights, sharpened_images):

    strength = edge_strength(img)

    print(f"Weight = {w}")
    print(f"- Edge Strength: {strength:.2f}")

    if w == 0.5:
        print("- Mild sharpening")
        print("- Slight enhancement of edges")
        print("- Natural appearance")
        print("- Minimal noise amplification\n")

    elif w == 1.0:
        print("- Moderate sharpening")
        print("- Clear edge enhancement")
        print("- Good balance between sharpness and image quality")
        print("- Suitable for most images\n")

    elif w == 1.5:
        print("- Strong sharpening")
        print("- Fine details become more visible")
        print("- Increased edge contrast")
        print("- Slight halo artifacts may appear\n")

    elif w == 2.0:
        print("- Very strong sharpening")
        print("- Maximum edge enhancement")
        print("- Fine details highly emphasized")
        print("- Noise amplification is noticeable")
        print("- Possible over-sharpening and halo effects\n")

# CONCLUSION

print(" CONCLUSION ")

print("""
Unsharp masking enhances image sharpness by adding a weighted
high-frequency mask to the original image.

OBSERVATIONS:

• Weight 0.5:
  - Produces mild sharpening.
  - Preserves a natural appearance.
  - Suitable when only slight enhancement is required.

• Weight 1.0:
  - Provides balanced sharpening.
  - Enhances edges clearly without excessive artifacts.
  - Gives good overall visual quality.

• Weight 1.5:
  - Produces strong sharpening.
  - Highlights fine image details.
  - May introduce minor halo effects around edges.

• Weight 2.0:
  - Produces very strong sharpening.
  - Maximizes edge visibility and detail enhancement.
  - Can amplify noise and create over-sharpening artifacts.

FINAL INFERENCE:

As the mask weight increases, image sharpness and edge strength
increase. However, excessive sharpening can introduce noise and
halo artifacts. Weight values between 1.0 and 1.5 generally
provide the best balance between edge enhancement and image quality,
while 2.0 demonstrates aggressive sharpening.
""")