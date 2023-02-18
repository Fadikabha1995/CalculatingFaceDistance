import cv2
import numpy as np

# Set up the camera
cap = cv2.VideoCapture(0)

# Set up the calibration parameters
FOCAL_LENGTH = 800  # in pixels
KNOWN_FACE_WIDTH = 0.14  # in meters

# Define a function to calculate the distance to a human face
def calculate_distance_to_face(face_width):
    # Calculate the distance to the face using the calibration parameters
    distance = (KNOWN_FACE_WIDTH * FOCAL_LENGTH) / face_width
    return distance

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Main loop
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    # Loop over the detected faces and draw a rectangle and distance text for each one
    for (x, y, w, h) in faces:
        # Calculate the distance to the face
        distance = calculate_distance_to_face(w)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Draw the distance text above the face
        distance_text = f"{distance:.2f} meters"
        cv2.putText(frame, distance_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with the distance text for each face
    cv2.imshow("Frame", frame)

    # Check for key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
