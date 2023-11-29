import cv2 
import numpy as np 
import pathlib

# Constants for gender detection model
GENDER_MODEL = './model/deploy_gender.prototxt'
GENDER_PROTO = './model/gender_net.caffemodel'
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['Male', 'Female']

# Constants for face, left ear, and right ear detection using Haar Cascade
LEFT_EAR_CASCADE_PATH = './Cascades/haarcascade_mcs_leftear.xml'
RIGHT_EAR_CASCADE_PATH = './Cascades/haarcascade_mcs_rightear.xml'
FACE_CASCADE_PATH = './Cascades/haarcascade_frontalface_default.xml'

# Load the pre-trained gender detection model
gender_model = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

# Load Haar Cascade Classifiers for face, left ear, and right ear detection
left_ear_cascade = cv2.CascadeClassifier(LEFT_EAR_CASCADE_PATH)
right_ear_cascade = cv2.CascadeClassifier(RIGHT_EAR_CASCADE_PATH)
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

if left_ear_cascade.empty() or right_ear_cascade.empty() or face_cascade.empty():
    raise IOError('Unable to load one of the cascade classifier XML files')

cap = cv2.VideoCapture(0)

# Set the desired window size
cv2.namedWindow('Ear, Face, and Gender Detector', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Ear, Face, and Gender Detector', 800, 600)  # Adjust the size as needed

scaling_factor = 0.5
while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect left ear
    left_ear = left_ear_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=3)
    for (x, y, w, h) in left_ear:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        cv2.putText(frame, "Left Ear", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Detect right ear
    right_ear = right_ear_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=3)
    for (x, y, w, h) in right_ear:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        cv2.putText(frame, "Right Ear", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Detect face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = frame[y:y + h, x:x + w]

        # Resize the face region to match the input size expected by the gender detection model
        blob = cv2.dnn.blobFromImage(face_roi, scalefactor=1.0, size=(227, 227), mean=MODEL_MEAN_VALUES, swapRB=False)
        gender_model.setInput(blob)
        gender_preds = gender_model.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]

        # Draw a rectangle around the face, display gender information, and display rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, f"Gender: {gender}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Ear, Face, and Gender Detector', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
