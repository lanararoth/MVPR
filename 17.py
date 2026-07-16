import cv2

# Load MPEG video
video = cv2.VideoCapture(r"C:\Users\shalu\Desktop\codes\MVPR\sample_960x540.mpeg")

if not video.isOpened():
    print("Error opening video file")
    exit()

while True:
    # Read frame
    ret, frame = video.read()

    if not ret:
        break

    # Convert to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert to Binary
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Display Frames
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Grayscale Frame", gray)
    cv2.imshow("Binary Frame", binary)

    # Press 'q' to quit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()