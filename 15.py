import cv2
import matplotlib.pyplot as plt

video = cv2.VideoCapture(r"C:\Users\shalu\Desktop\codes\MVPR\video1.mpeg")

frame_no = 0

while True:
    ret, frame = video.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(f"Original {frame_no}")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(binary, cmap="gray")
    plt.title("Binary")
    plt.axis("off")

    plt.show()

    frame_no += 1

    # Display every 30th frame (change as needed)
    if frame_no % 30 != 0:
        continue

video.release()