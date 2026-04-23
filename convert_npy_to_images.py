import os
import numpy as np
import cv2


PROCESSED_DIR = "processed"     
DATASET_DIR = "dataset"         


for split in ["train", "test"]:
    for cls in ["real", "fake"]:
        os.makedirs(os.path.join(DATASET_DIR, split, cls), exist_ok=True)


def save_images(label):
    src_dir = os.path.join(PROCESSED_DIR, label)
    files = [f for f in os.listdir(src_dir) if f.endswith(".npy")]

    print(f"[INFO] Converting {label} .npy → images ({len(files)} files)")

    for i, file in enumerate(files):
        arr = np.load(os.path.join(src_dir, file))

        img = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

     
        split = "train" if i % 5 != 0 else "test"

        out_path = os.path.join(DATASET_DIR, split, label, file.replace(".npy", ".jpg"))
        cv2.imwrite(out_path, img)



save_images("real")
save_images("fake")

print("✔ All .npy files converted to images successfully!")
