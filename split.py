import os, shutil
from sklearn.model_selection import train_test_split

BASE = "processed"
DATASET = "dataset"

classes = ["real", "fake"]

for split in ["train", "test"]:
    for cls in classes:
        os.makedirs(f"{DATASET}/{split}/{cls}", exist_ok=True)

for cls in classes:
    files = os.listdir(f"{BASE}/{cls}")
    files = [f for f in files if f.endswith(".npy")]

    train, test = train_test_split(files, test_size=0.2, random_state=42)

    for f in train:
        shutil.copy(f"{BASE}/{cls}/{f}", f"{DATASET}/train/{cls}/{f}")

    for f in test:
        shutil.copy(f"{BASE}/{cls}/{f}", f"{DATASET}/test/{cls}/{f}")

print("Split complete!")
