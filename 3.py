import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r"C:\Users\shalu\Desktop\codes\MVPR\image.jpg")

if image is None:
    print("Error loading image")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# GAUSSIAN NOISE

gaussian_noise = np.random.normal(0, 25, image_rgb.shape)
gaussian_noisy = image_rgb + gaussian_noise
gaussian_noisy = np.clip(gaussian_noisy, 0, 255).astype(np.uint8)

# SALT & PEPPER NOISE

sp_noisy = image_rgb.copy()

prob = 0.02  # noise density

# Salt noise (white pixels)
salt = np.random.rand(*image_rgb.shape[:2]) < prob
sp_noisy[salt] = 255

# Pepper noise (black pixels)
pepper = np.random.rand(*image_rgb.shape[:2]) < prob
sp_noisy[pepper] = 0

# AVERAGE FILTERING (on both noise images)

# Gaussian
g_3x3 = cv2.blur(gaussian_noisy, (3,3))
g_5x5 = cv2.blur(gaussian_noisy, (5,5))
g_7x7 = cv2.blur(gaussian_noisy, (7,7))

# Salt & pepper
sp_3x3 = cv2.blur(sp_noisy, (3,3))
sp_5x5 = cv2.blur(sp_noisy, (5,5))
sp_7x7 = cv2.blur(sp_noisy, (7,7))



fig, axes = plt.subplots(2, 4, figsize=(18, 8))

# Row 1 - Gaussian noise
axes[0,0].imshow(gaussian_noisy)
axes[0,0].set_title("Gaussian Noise")
axes[0,0].axis("off")

axes[0,1].imshow(g_3x3)
axes[0,1].set_title("3×3 Filter")
axes[0,1].axis("off")

axes[0,2].imshow(g_5x5)
axes[0,2].set_title("5×5 Filter")
axes[0,2].axis("off")

axes[0,3].imshow(g_7x7)
axes[0,3].set_title("7×7 Filter")
axes[0,3].axis("off")

# Row 2 - Salt & Pepper noise
axes[1,0].imshow(sp_noisy)
axes[1,0].set_title("Salt & Pepper Noise")
axes[1,0].axis("off")

axes[1,1].imshow(sp_3x3)
axes[1,1].set_title("3×3 Filter")
axes[1,1].axis("off")

axes[1,2].imshow(sp_5x5)
axes[1,2].set_title("5×5 Filter")
axes[1,2].axis("off")

axes[1,3].imshow(sp_7x7)
axes[1,3].set_title("7×7 Filter")
axes[1,3].axis("off")

plt.tight_layout()
plt.show()

# IMAGE ANALYSIS

def estimate_smoothness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def noise_level(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.std(gray)

metrics = {
    "Gaussian Noisy": (estimate_smoothness(gaussian_noisy), noise_level(gaussian_noisy)),
    "Gaussian 3x3": (estimate_smoothness(g_3x3), noise_level(g_3x3)),
    "Gaussian 5x5": (estimate_smoothness(g_5x5), noise_level(g_5x5)),
    "Gaussian 7x7": (estimate_smoothness(g_7x7), noise_level(g_7x7)),

    "Salt & Pepper Noisy": (estimate_smoothness(sp_noisy), noise_level(sp_noisy)),
    "SP 3x3": (estimate_smoothness(sp_3x3), noise_level(sp_3x3)),
    "SP 5x5": (estimate_smoothness(sp_5x5), noise_level(sp_5x5)),
    "SP 7x7": (estimate_smoothness(sp_7x7), noise_level(sp_7x7)),
}

print("\nFILTER ANALYSIS\n")

for name, (sharpness, noise) in metrics.items():
    print(f"{name}")
    print(f"- Sharpness (Laplacian Variance): {sharpness:.2f}")
    print(f"- Noise Level (Std Dev): {noise:.2f}\n")

print("\nCONCLUSION")
print("""
1. Gaussian Noise:
   - Mean filter reduces noise effectively
   - Larger kernels increase smoothing but blur details

2. Salt & Pepper Noise:
   - Mean filter is less effective compared to Gaussian noise
   - Larger kernels remove noise but heavily blur edges

FINAL INFERENCE:
Mean filtering is suitable for Gaussian noise but not optimal for Salt & Pepper noise.
For better results, median filtering is recommended for impulse noise.
""")