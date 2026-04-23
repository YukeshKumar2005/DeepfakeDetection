[ Deepfake Detection Project 🎭 | CNN + Transfer Learning
This project focuses on detecting real vs fake (deepfake) videos using both a custom CNN model and a Transfer Learning model (MobileNetV2).
It includes preprocessing scripts, training pipelines, evaluation metrics, EDA visualizations, and prediction tools for images and videos.

 Project Structure
project/
│── data/                     # Raw dataset (FaceForensics++)  
│   ├── real/
│   └── fake/
│
│── processed/                # Preprocessed images (224×224)
│
│── cnn_model/                # CNN model files (if created)
│
│── transfer_model.h5         # Final Transfer Learning model (MobileNetV2)
│
│── preprocess.py             # Extract frames + resize images
│── train_transfer.py         # Training transfer learning model
│── test_video.py             # Predict deepfake from video
│── predict_image.py          # Predict deepfake from single image
│── evaluate.py               # Accuracy, classification report, confusion matrix
│── eda.py                    # EDA: FPS, resolution, sample frames
│
│── README.md

📦 Dataset
This project uses the FaceForensics++ dataset.

Each video is labeled as:

REAL

FAKE (Deepfake)

During preprocessing, frames are extracted and stored for training/testing.

🛠️ Installation
1️⃣ Install Requirements
pip install tensorflow opencv-python scikit-learn matplotlib numpy
2️⃣ Verify GPU (optional)
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
🔧 Preprocessing
Extract frames from each video and save them in processed/:

python preprocess.py
This script:

✔ Loads videos
✔ Extracts frames
✔ Resizes to 224×224
✔ Saves into train/test folders

🧠 Model Training
Train the Transfer Learning (MobileNetV2) model:

python train_transfer.py
Outputs:

transfer_model.h5

Training accuracy

Validation accuracy

🧪 Model Evaluation
Generate accuracy, classification report, and confusion matrix:

python evaluate.py
Example output:

Accuracy: 95.12%

Confusion Matrix:
[[150   4]
 [ 13 182]]

Classification Report:
FAKE  – Precision 0.92 | Recall 0.97
REAL  – Precision 0.98 | Recall 0.93
🖼️ Predict From Image
python predict_image.py --path sample.jpg
Output:

Prediction: FAKE
Confidence: 87%
🎬 Predict From Video
python test_video.py --video input.mp4
This script:

✔ Extracts frames
✔ Predicts each frame
✔ Gives final decision (Majority Vote)

Example:

Final Video Prediction: REAL (82% frames real)
📊 Exploratory Data Analysis (EDA)
Run:

python eda.py
Generates:

📌 Class distribution (REAL vs FAKE)
📌 Resolution distribution
📌 FPS distribution
📌 Sample extracted frames

Outputs saved in: eda_outputs/

📈 Results
Achieved 95% accuracy using MobileNetV2 Transfer Learning.

Confusion matrix shows strong generalization on unseen data.

💡 Future Work
Add LSTM/3D CNN for temporal modeling

Use ViT (Vision Transformer)

Deploy model with Flask / FastAPI

Create real-time webcam deepfake detector

# 🎭 FaceForensics++ (C23) Deepfake Detection Project

A deepfake detection project using the **FaceForensics++ (C23)** dataset.  
This project focuses on detecting manipulated videos using machine learning / deep learning techniques.

---

# 📌 About the Dataset

This project uses the **FaceForensics++ (C23)** dataset sourced from Kaggle.

⚠️ The dataset videos are **not included** in this repository because:
- the dataset is very large
- GitHub has storage limits
- keeping the repository lightweight and fast is better practice

You can download the dataset from Kaggle using the link below.

Dataset Link:
https://www.kaggle.com/code/xdxd003/download-ff-mega
