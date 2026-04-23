import os
import cv2
import matplotlib.pyplot as plt

DATASET_PATH = r"C:\supervised project\sample\archive\project\data"

real_count = len(os.listdir(os.path.join(DATASET_PATH, "original")))
fake_count = len(os.listdir(os.path.join(DATASET_PATH, "DeepFakeDetection")))

print("Real videos:", real_count)
print("Fake videos:", fake_count)

plt.bar(["REAL", "FAKE"], [real_count, fake_count], color=["green", "red"])
plt.title("Class Distribution")
plt.ylabel("Number of videos")
plt.show()



widths = []
heights = []

for cls in ["original", "DeepFakeDetection"]:
    class_path = os.path.join(DATASET_PATH, cls)

    for vid in os.listdir(class_path):
        if not vid.endswith(".mp4"):
            continue

        path = os.path.join(class_path, vid)
        cap = cv2.VideoCapture(path)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        widths.append(w)
        heights.append(h)

        cap.release()

plt.scatter(widths, heights, alpha=0.6)
plt.title("Resolution Distribution")
plt.xlabel("Width")
plt.ylabel("Height")
plt.show()
fps_values = []

for cls in ["original", "DeepFakeDetection"]:
    class_path = os.path.join(DATASET_PATH, cls)

    for vid in os.listdir(class_path):
        if not vid.endswith(".mp4"):
            continue

        path = os.path.join(class_path, vid)
        cap = cv2.VideoCapture(path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps_values.append(fps)

        cap.release()

plt.hist(fps_values, bins=10, color="blue", alpha=0.7)
plt.title("FPS Distribution")
plt.xlabel("FPS")
plt.ylabel("Count")
plt.show() 


def show_sample_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Could not read frame.")
        return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame)
    plt.title(os.path.basename(video_path))
    plt.axis("off")
    plt.show()


real_video = os.path.join(DATASET_PATH, "original", os.listdir(os.path.join(DATASET_PATH, "original"))[0])
fake_video = os.path.join(DATASET_PATH, "DeepFakeDetection", os.listdir(os.path.join(DATASET_PATH, "DeepFakeDetection"))[0])

show_sample_frame(real_video)
show_sample_frame(fake_video)