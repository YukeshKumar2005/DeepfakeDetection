import numpy as np
import tensorflow as tf
import os
import cv2

MODEL_PATH = "transfer_model.h5"
TEST_DIR = "dataset/test"
IMG_SIZE = 224

model = tf.keras.models.load_model(MODEL_PATH)

def load_folder(folder, label):
    X, y = [], []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        X.append(img)
        y.append(label)
    return X, y

X_real, y_real = load_folder(os.path.join(TEST_DIR, "real"), 0)
X_fake, y_fake = load_folder(os.path.join(TEST_DIR, "fake"), 1)

X = np.array(X_real + X_fake)
y_true = np.array(y_real + y_fake)

preds = model.predict(X)
y_pred = (preds > 0.5).astype(int).flatten()

accuracy = np.mean(y_pred == y_true)
print("Accuracy:", accuracy * 100, "%")
