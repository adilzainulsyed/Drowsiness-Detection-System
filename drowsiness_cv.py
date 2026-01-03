import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import serial
import time

# Load model
model = load_model("eye_state_cnn.h5")

# Arduino Serial
arduino = serial.Serial('COM3', 9600)  # Change COM port
time.sleep(2)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)
IMG_SIZE = 64
closed_frames = 0
THRESHOLD = 20

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def get_eye(frame, landmarks, eye_points):
    h, w, _ = frame.shape
    points = [(int(landmarks[p].x*w), int(landmarks[p].y*h)) for p in eye_points]
    x, y, w, h = cv2.boundingRect(np.array(points))
    eye = frame[y:y+h, x:x+w]
    return eye

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status = "AWAKE"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye = get_eye(frame, face_landmarks.landmark, LEFT_EYE)
            right_eye = get_eye(frame, face_landmarks.landmark, RIGHT_EYE)

            if left_eye.size == 0 or right_eye.size == 0:
                continue

            left_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
            right_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)

            left_eye = cv2.resize(left_eye, (IMG_SIZE, IMG_SIZE)) / 255.0
            right_eye = cv2.resize(right_eye, (IMG_SIZE, IMG_SIZE)) / 255.0

            left_eye = left_eye.reshape(1, IMG_SIZE, IMG_SIZE, 1)
            right_eye = right_eye.reshape(1, IMG_SIZE, IMG_SIZE, 1)

            left_pred = model.predict(left_eye)[0][0]
            right_pred = model.predict(right_eye)[0][0]

            if left_pred < 0.5 and right_pred < 0.5:
                closed_frames += 1
            else:
                closed_frames = 0

            if closed_frames >= THRESHOLD:
                status = "SLEEPING"
                arduino.write(b'1')
            else:
                arduino.write(b'0')

    cv2.putText(frame, status, (30,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,0,255) if status=="SLEEPING" else (0,255,0), 2)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()