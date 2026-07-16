import cv2
import matplotlib.pyplot as plt

# Load two grayscale images
img1 = cv2.imread(r"C:\Users\shalu\Desktop\codes\MVPR\image.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(r"C:\Users\shalu\Desktop\codes\MVPR\grayscale.jpg", cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Error: Could not load one or both images.")
    exit()

# Create SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Create Brute-Force Matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)

# Sort matches based on distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 50 matches
matched_image = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    matches[:50],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Display matched image
plt.figure(figsize=(15,8))
plt.imshow(matched_image, cmap='gray')
plt.title("SIFT Feature Matching using Brute-Force Matcher")
plt.axis("off")
plt.show()