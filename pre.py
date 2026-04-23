import cv2
import os
import numpy as np
from tqdm import tqdm

ORIGINAL_DIR = "data/original"
FAKE_DIR = "data/DeepFakeDetection"

OUTPUT_REAL = "processed/real"
OUTPUT_FAKE = "processed/fake"

os.makedirs(OUTPUT_REAL, exist_ok=True)
(os.makedirs(OUTPUT_FAKE, exist_ok=True))

haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(haar_path)

def extract_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return frame[y:y+h, x:x+w]

def process_videos(folder, output):
    videos = [v for v in os.listdir(folder) if v.endswith(".mp4")]
    for vid in tqdm(videos, desc=f"Processing {folder}"):
        path = os.path.join(folder, vid)
        cap = cv2.VideoCapture(path)

        ret, frame = cap.read()
        if not ret:
            continue

        face = extract_face(frame)
        if face is None:
            continue

        face = cv2.resize(face, (224, 224))
        face = face.astype("float32") / 255.0

        np.save(os.path.join(output, vid.replace(".mp4", ".npy")), face)

        cap.release()

print("Processing REAL videos...")
process_videos(ORIGINAL_DIR, OUTPUT_REAL)

print("Processing FAKE videos...")
process_videos(FAKE_DIR, OUTPUT_FAKE)

print("Done!")
