import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Load your trained gender detection model
left_ear_cascade = cv2.CascadeClassifier('./Cascades/haarcascade_mcs_leftear.xml')
right_ear_cascade = cv2.CascadeClassifier('./Cascades/haarcascade_mcs_rightear.xml')
face_cascade = cv2.CascadeClassifier('./Cascades/haarcascade_frontalface_default.xml')
gender_model = load_model('gender_model.h5')

cap = cv2.VideoCapture(0)  # Use 0 if you have only one camera
# Set the desired window size
cv2.namedWindow('Face,ears,gender Detector', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face,ears,gender Detector', 800, 600)  # Adjust the size as needed

scaling_factor = 0.5
while True:
    ret,frame = cap.read()
    frame = cv2.resize(frame, dsize=None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect ears
    left_ear = left_ear_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=3)
    right_ear = right_ear_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=3)

    for (x, y, w, h) in left_ear:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        
        # Extract the face region for the left ear
        face_roi = frame[y:y + h, x:x + w]

        # Resize the face region to match the input size expected by the gender detection model
        blob = cv2.dnn.blobFromImage(face_roi, scalefactor=1.0, size=(100, 100), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=True)
        
        # Manually reorder the channels (from OpenCV's default BGR to model's expected RGB)
        blob = np.transpose(blob, (0, 2, 3, 1))

        gender_preds = gender_model.predict(blob)
        gender = 'Male' if gender_preds[0][0] > 0.5 else 'Female'

        # Display gender information for the left ear
        cv2.putText(frame, f"Left Ear: Gender - {gender}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    for (x, y, w, h) in right_ear:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        # Extract the face region for the right ear
        face_roi = frame[y:y + h, x:x + w]

        # Resize the face region to match the input size expected by the gender detection model
        blob = cv2.dnn.blobFromImage(face_roi, scalefactor=1.0, size=(100, 100), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=True)
        
        # Manually reorder the channels (from OpenCV's default BGR to model's expected RGB)
        blob = np.transpose(blob, (0, 2, 3, 1))

        gender_preds = gender_model.predict(blob)
        gender = 'Male' if gender_preds[0][0] > 0.5 else 'Female'

        # Display gender information for the right ear
        cv2.putText(frame, f"Right Ear: Gender - {gender}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Detect face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=8)
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face, display gender information, and display rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame,"Face",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
        
    cv2.imshow('Ear,Face,Gender Detector', frame)
    c = cv2.waitKey(1)
    if c == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

