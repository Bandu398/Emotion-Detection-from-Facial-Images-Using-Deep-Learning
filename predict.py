import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ==========================
# Load Model
# ==========================
model = load_model("model/emotion_model.h5")

labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# ==========================
# Face Detector
# ==========================
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ==========================
# Open Webcam
# ==========================
cap = None

for camera in [0, 1, 2]:
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

    if cap.isOpened():
        print("Camera Found :", camera)
        break

if cap is None or not cap.isOpened():
    print("No webcam found.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press Q to Exit")

# ==========================
# Start Detection
# ==========================
while True:

    ret, frame = cap.read()

    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(40,40)
    )

    for (x,y,w,h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face,(48,48))

        face = face.astype("float32")/255.0
        face = np.expand_dims(face,-1)
        face = np.expand_dims(face,0)

        prediction = model.predict(face,verbose=0)

        emotion = labels[np.argmax(prediction)]

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.putText(
            frame,
            emotion,
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()