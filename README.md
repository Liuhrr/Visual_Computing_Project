# 🕺 Visual Computing Project - Just Dance Score System

A real-time dance scoring application that uses **YOLOv8-pose** for human pose estimation. 
It compares your webcam movements against a reference dance video and provides instant feedback with scores like **Perfect!**, **Super!**, **Good**, or **Miss**.

## ✨ Features

- **Real-time pose detection** using YOLOv8 (17 COCO keypoints)
- **Dance video playback** with skeleton overlay
- **Live webcam capture** with real-time pose visualization
- **Automatic scoring** by comparing joint angles between reference video and webcam
- **Instant visual feedback** with score percentage and performance labels
- **Tkinter-based GUI** with dual-panel display (video + webcam)

## 🛠️ Tech Stack

- Python 3.10+
- Ultralytics YOLOv8 (pose estimation)
- OpenCV (video/camera processing)
- Tkinter (GUI)
- NumPy / SciPy (angle calculation & similarity metrics)

## 🚀 Quick Start

```bash
git clone https://github.com/Liuhrr/Visual_Computing_Project.git
cd Visual_Computing_Project
pip install -r requirements.txt
python danceapp.py
