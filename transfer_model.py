import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os

DATASET_DIR = "dataset"


train_gen = ImageDataGenerator(rescale=1/255.).flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)

test_gen = ImageDataGenerator(rescale=1/255.).flow_from_directory(
    os.path.join(DATASET_DIR, "test"),
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)


base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False 


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print(model.summary())


history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=8
)

model.save("transfer_model.h5")
print("✔ Transfer Learning model saved as transfer_model.h5")
