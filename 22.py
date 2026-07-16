import cv2
import matplotlib.pyplot as plt

# Load grayscale images
img1 = cv2.imread(r"C:\Users\shalu\Desktop\codes\MVPR\image.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(r"C:\Users\shalu\Desktop\codes\MVPR\grayscale.jpg", cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Error loading images.")
    exit()

# Create SURF detector
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)

# Detect keypoints and compute descriptors
kp1, des1 = surf.detectAndCompute(img1, None)
kp2, des2 = surf.detectAndCompute(img2, None)

# Draw keypoints
img1_kp = cv2.drawKeypoints(
    img1, kp1, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

img2_kp = cv2.drawKeypoints(
    img2, kp2, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Display results
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(img1_kp, cmap='gray')
plt.title("SURF Keypoints - Image 1")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(img2_kp, cmap='gray')
plt.title("SURF Keypoints - Image 2")
plt.axis("off")

plt.tight_layout()
plt.show()

# Print information
print("Image 1")
print("Number of Keypoints:", len(kp1))
print("Descriptor Shape:", des1.shape)

print("\nImage 2")
print("Number of Keypoints:", len(kp2))
print("Descriptor Shape:", des2.shape)