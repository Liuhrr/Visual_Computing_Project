# 🕺 Visual Computing Project - Just Dance Score System

A real-time dance scoring application that uses **YOLOv8-pose** for human pose estimation. 
It compares your webcam movements against a reference dance video and provides instant feedback with scores like **Perfect!**, **Super!**, **Good**, or **Miss**.

---

## ✨ Features

- **Real-time pose detection** using YOLOv8 (17 COCO keypoints)
- **Dance video playback** with skeleton overlay
- **Live webcam capture** with real-time pose visualization
- **Automatic scoring** by comparing joint angles between reference video and webcam
- **Instant visual feedback** with score percentage and performance labels
- **Tkinter-based GUI** with dual-panel display (video + webcam)

---

## 🛠️ Tech Stack

- Python 3.10+
- Ultralytics YOLOv8 (pose estimation)
- OpenCV (video/camera processing)
- Tkinter (GUI)
- NumPy / SciPy (angle calculation & similarity metrics)

---

## 🚀 Quick Start

### 1. Clone the repository
git clone https://github.com/Liuhrr/Visual_Computing_Project.git
cd Visual_Computing_Project

### 2. Install dependencies
pip install -r requirements.txt

### 3. Download the YOLOv8 model
The model will be automatically downloaded when you first run the program, or you can manually place `yolov8n-pose.pt` in the `models/` folder.

### 4. Run the application
python danceapp.py

---

## 📂 Project Structure
Visual_Computing_Project/
├── danceapp.py              # Main GUI application
├── models/
│   └── yolov8n-pose.pt      # YOLOv8 pose estimation model
├── utils/
│   ├── __init__.py
│   ├── pose_utils.py        # Keypoint extraction & angle calculation
│   └── scoring.py           # Similarity & scoring logic
├── data/                    # Reference videos (local only)
│   └── TikTokDataset/       # Sample dance videos
├── outputs/                 # Saved results & logs
├── .gitignore               # Ignored files (videos, cache, etc.)
└── requirements.txt         # Python dependencies

---

## 📊 How Scoring Works

1. **Pre-extract** joint angles from every frame of the reference video
2. **Extract** joint angles from your webcam feed in real-time
3. **Compare** angle vectors frame-by-frame
4. **Map** similarity to a score and feedback label

| Score Range | Feedback |
|-------------|----------|
| ≥ 90%       | Perfect! |
| 75% – 89%   | Super!   |
| 60% – 74%   | Good     |
| < 60%       | Miss     |

---

## 🎮 How to Use

1. Launch the GUI: `python danceapp.py`
2. Click **Open Video** to select a reference dance video (MP4)
3. Click **Load Reference & Score** to pre-process the video
4. Click **Start Webcam** to activate your camera
5. Perform the dance moves in front of the camera
6. The score and feedback will update in real-time

---

## 📝 Notes

- Video files (`.mp4`) are **excluded** from the repository via `.gitignore` to keep it lightweight.
- Place your own reference videos in `data/` for testing.
- Make sure your webcam is available and not occupied by other applications.
- The scoring system uses **frame-by-frame alignment** – try to match the rhythm of the reference video for best results.

---

## 📎 Acknowledgments

- Built for the Visual Computing course project
- Uses COCO dataset keypoint format (17 points)
- YOLOv8-pose model by Ultralytics


