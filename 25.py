from ultralytics import YOLO
import cv2

# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")   # Nano model

# Image path
image_path = "test.jpg"

# Perform detection
results = model(image_path)

# Get annotated image
annotated = results[0].plot()

# Display image
cv2.imshow("YOLOv8 Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save output
cv2.imwrite("detected_image.jpg", annotated)

print("Detection completed.")
