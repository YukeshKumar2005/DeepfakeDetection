import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


MODEL_PATH = "transfer_model.h5"
TEST_DIR = "dataset/test"
IMG_SIZE = 224
BATCH_SIZE = 32


model = tf.keras.models.load_model(MODEL_PATH)
print("Model Loaded Successfully!")



test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("\nClass Mapping:")
print(test_data.class_indices)  


pred_probs = model.predict(test_data)
pred_labels = (pred_probs > 0.5).astype(int).flatten()

true_labels = test_data.classes


print("\nAccuracy:", accuracy_score(true_labels, pred_labels))

print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=["FAKE", "REAL"]))

print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, pred_labels))
