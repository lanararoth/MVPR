import cv2
import matplotlib.pyplot as plt

# Capture image from Webcam

cap = cv2.VideoCapture(0)

print("Press 's' to capture image")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to access webcam")
        break

    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)

    # Save image
    if key == ord('s'):
        cv2.imwrite("captured_image.jpg", frame)
        print("Image captured and saved!")
        break

    # Quit without saving
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Load image from Dataset

image = cv2.imread(r"C:\Users\shalu\Desktop\codes\MVPR\image.jpg")

# if image loaded properly oor not

if image is None:
    print("Error loading image")
    exit()

# IMAGE PREPROCESSING


# Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Histogram Equalization
equalized = cv2.equalizeHist(gray)

# DISPLAY RESULTS

plt.figure(figsize=(12,4))

# Original Image
plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

# Grayscale Image
plt.subplot(1,3,2)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis("off")

# Histogram Equalized Image
plt.subplot(1,3,3)
plt.imshow(equalized, cmap='gray')
plt.title("Histogram Equalization")
plt.axis("off")

plt.show()