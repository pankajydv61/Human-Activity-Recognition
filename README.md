# 🎥 Human Activity Recognition using Deep Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.16-orange?style=for-the-badge&logo=tensorflow"/>
  <img src="https://img.shields.io/badge/MediaPipe-0.10-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-red?style=for-the-badge&logo=opencv"/>
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit"/>
</p>

<p align="center">
A Deep Learning based Human Activity Recognition system that classifies human activities from video using MediaPipe Pose Estimation and TensorFlow.
</p>

---

# 📌 Overview

Human Activity Recognition (HAR) is an Artificial Intelligence application that identifies and classifies human actions from videos.

This project extracts **33 body pose landmarks** using **MediaPipe**, converts them into temporal sequences, and feeds them into a **TensorFlow deep learning model** to predict the activity being performed.

The application provides an interactive **Streamlit web interface** where users can upload videos and receive activity predictions with confidence scores.

---

# ✨ Features

✅ Upload video files for prediction

✅ Human pose extraction using MediaPipe

✅ Deep Learning based activity classification

✅ Interactive Streamlit Web Interface

✅ Confidence score for each prediction

✅ Dataset preprocessing pipeline

✅ Model training pipeline

✅ Easy-to-use prediction system

---

# 🏋️ Supported Activities

- 🏀 Basketball
- 🤸 Jumping Jack
- 💪 Pull Ups
- 🥊 Boxing Speed Bag
- 👊 Punch
- 🏌 Golf Swing
- 🚶 Walking With Dog
- 🧘 Tai Chi
- 🏋 Body Weight Squats
- 💪 Push Ups

---

# 🧠 Project Architecture

```
Video Input
      │
      ▼
MediaPipe Pose Detection
      │
      ▼
Pose Landmark Extraction
      │
      ▼
Sequence Generation
      │
      ▼
TensorFlow Deep Learning Model
      │
      ▼
Predicted Human Activity
```

---

# 📂 Project Structure

```
Human-Activity-Recognition/

│
├── dataset/
│   ├── raw/
│   ├── processed/
│   └── uploads/
│
├── models/
│   └── activity_model.keras
│
├── utils/
│   └── mediapipe_utils.py
│
├── app.py
├── train.py
├── predict.py
├── prepare_dataset.py
├── extract_keypoints.py
├── config.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Technologies Used

| Category | Technology |
|-----------|------------|
| Language | Python |
| Deep Learning | TensorFlow / Keras |
| Pose Estimation | MediaPipe |
| Computer Vision | OpenCV |
| Web Framework | Streamlit |
| Numerical Computing | NumPy |
| Machine Learning | Scikit-Learn |

---

# 📊 Model Performance

| Metric | Value |
|---------|------:|
| Activities | 10 |
| Dataset Samples | 1227 |
| Sequence Length | 30 Frames |
| Pose Features | 132 |
| Test Accuracy | **82.52%** |

---

# 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/pankajydv61/Human-Activity-Recognition.git
```

### Navigate

```bash
cd Human-Activity-Recognition
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open at

```
http://localhost:8501
```

---

# 🎯 Workflow

1. Upload a video.
2. MediaPipe extracts body landmarks.
3. Pose sequences are generated.
4. TensorFlow model predicts activity.
5. Confidence score is displayed.

---

# 📸 Screenshots

> Add screenshots here.

Example:

```
screenshots/
│
├── home.png
├── upload.png
├── prediction.png
```

---

# 📈 Future Improvements

- Real-time Webcam Detection
- Live Pose Skeleton Visualization
- Improved Deep Learning Architecture
- More Activity Classes
- Higher Model Accuracy
- Docker Deployment
- REST API using FastAPI
- Interactive React Frontend
- Mobile-Friendly Interface

---

# 👨‍💻 Author

**Pankaj Yadav**

MCA (Data Science)

Lovely Professional University

GitHub:
https://github.com/pankajydv61

LinkedIn:
https://www.linkedin.com/in/pankajyadav61/

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

It motivates me to build more AI and Machine Learning projects.

---

# 📜 License

This project is developed for educational and research purposes.
