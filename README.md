# 😊 Face Emotion Detector with Songs

**Real‑time emotion detection from facial expressions | CNN‑based model | Mood‑based music playback**

Face Emotion Detector is a real‑time computer vision system that uses your webcam to detect your facial expression, predicts your emotion (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise), and automatically plays a matching song. Built with OpenCV and a pre‑trained Keras CNN model.

---

## ✨ Features

- **Real‑time Face Detection** – Uses Haar Cascade to locate faces in the webcam feed.
- **Emotion Recognition** – A CNN model (trained on FER2013) classifies 7 emotions with.
- **Mood‑Based Music Playback** – Plays a local audio file that matches the detected emotion (e.g., `happy.mp3` for happy).
- **Live Video Overlay** – Draws a bounding box around your face and displays the predicted emotion.
- **Frame Smoothing** – Averages predictions over multiple frames to reduce flickering and neutral bias.

---

## 🧠 Model Details

- **Architecture:** Custom CNN (2 convolutional layers + dense layers)
- **Training dataset:** FER2013 (35,887 grayscale 48×48 face images)
- **Emotion classes:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Input:** Grayscale 48×48 pixel face ROI
- **File size:** 48.5 MB (`facialemotionmodel.h5`)

---

## 🚀 How to Run Locally

### Prerequisites

- Python 3.7 or higher
- A working webcam
- Git (to clone the repository)

### Step 1: Clone the repository

```bash
git clone https://github.com/saadf3819/Face-emotion-detector-with-songs.git
cd Face-emotion-detector-with-songs
