import cv2
import numpy as np
import os

# -----------------------------
#  Load Classifier + Haarcascade
# -----------------------------
if not os.path.exists("Face-Detection\classifier.xml"):
    print("Error: classifier.xml not found!")
    exit()

clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("Face-Detection\classifier.xml")

faceCascade = cv2.CascadeClassifier("Face-Detection\haarcascade_frontalface_default.xml")

if faceCascade.empty():
    print("Error: Haarcascade XML file not found!")
    exit()

# -----------------------------
#  Name Mapping (ID → Name)
# -----------------------------
names = {
    1: "Zihadul Talukder",
    2: "Prince Sarker",
    3: "User 3"
}

# -----------------------------
#  Start Webcam
# -----------------------------
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open camera")
    exit()

print("Camera started. Press 'q' to quit.")

while True:
    ret, frame = cam.read()
    if not ret or frame is None:
        print("Warning: No frame captured")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    for (x, y, w, h) in faces:

        # Predict face ID
        face_id, confidence = clf.predict(gray[y:y+h, x:x+w])

        # Confidence value (lower = better)
        if confidence < 80:
            person_name = names.get(face_id, "Unknown")
        else:
            person_name = "Unknown"

        # Draw box + name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{person_name} ({int(confidence)})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    # Press 'q' to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
