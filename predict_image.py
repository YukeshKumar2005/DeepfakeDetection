import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "transfer_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 224

def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        print("Error: image not found!")
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def predict_image(image_path):
    img = preprocess_image(image_path)
    if img is None:
        return

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        print("Prediction: FAKE")
    else:
        print("Prediction: REAL")

if __name__ == "__main__":
    test_img = r"C:\supervised project\sample\archive\project\dataset\test\real\996.jpg"  # <-- change path if needed
    predict_image(test_img)
