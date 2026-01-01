import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# =========================
# MODEL ARCHITECTURE
# =========================
class EyeCNN(nn.Module):
    def __init__(self):
        super(EyeCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(64 * 11 * 11, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# LOAD MODEL
# =========================
model = EyeCNN().to(device)
model.load_state_dict(torch.load("eye_state_cnn.pt", map_location=device))
model.eval()
print("Model loaded")

classes = ["closed", "open"]

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((90, 90)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# =========================
# LOAD HAAR CASCADES
# =========================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# =========================
# WEBCAM LOOP
# =========================
cap = cv2.VideoCapture(0)
print("Press q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---------- FACE DETECTION ----------
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]

        # ---------- EYE DETECTION ----------
        eyes = eye_cascade.detectMultiScale(face_roi)

        # sort eyes left→right then pick RIGHT EYE only
        eyes = sorted(eyes, key=lambda e: e[0])  # sort by x
        if len(eyes) >= 2:
            right_eye = eyes[1]  # index 1 = right eye from camera view
        elif len(eyes) == 1:
            right_eye = eyes[0]
        else:
            continue

        (ex, ey, ew, eh) = right_eye

        # absolute coords
        rx, ry, rw, rh = x + ex, y + ey, ew, eh

        # draw tracking box
        cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)

        # crop eye
        eye_img = gray[ry:ry+rh, rx:rx+rw]

        # preprocess
        eye_resized = cv2.resize(eye_img, (90, 90))
        pil_img = Image.fromarray(eye_resized)
        tensor = transform(pil_img).unsqueeze(0).to(device)

        # ---------- PREDICT ----------
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)
            pred = torch.argmax(probs)

        label = classes[pred]
        p = probs[0][pred].item()

        # put label above box
        cv2.putText(
            frame,
            f"Right Eye: {label} ({p:.2f})",
            (rx, ry-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Right Eye Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
