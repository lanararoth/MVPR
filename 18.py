import cv2
import numpy as np

# Load video
video = cv2.VideoCapture(r"C:\Users\shalu\Desktop\codes\MVPR\sample_960x540.mpeg")

if not video.isOpened():
    print("Error opening video file")
    exit()

# Create Background Subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=False
)

# Morphological kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Threshold to remove gray pixels
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # Morphological Operations
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes
    for contour in contours:
        if cv2.contourArea(contour) > 500:   # Ignore small objects
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h),
                          (0, 255, 0), 2)

    # Display outputs
    cv2.imshow("Original Video", frame)
    cv2.imshow("Foreground Mask", fg_mask)

    # Exit on pressing 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()