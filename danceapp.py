import sys
import os
import cv2
import numpy as np
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

# Import utilities
from utils.pose_utils import (
    extract_keypoints_from_result,
    extract_angle_vector,
    ANGLE_TRIPLETS
)
from utils.scoring import compute_angle_similarity, map_score_to_feedback

# Load YOLOv8 pose model
model = YOLO("models/yolov8n-pose.pt")

# COCO skeleton connections (keypoint pairs)
skeleton = [
    (0, 5), (0, 6),  # nose to shoulders
    (5, 6),          # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10), # right arm
    (5, 11), (6, 12),# torso sides
    (11, 12),        # hips
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16) # right leg
]

class PoseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stickman Dance GUI")
        self.root.geometry("1200x700")   # Slightly taller to fit scoring labels

        self.running_file = False
        self.running_cam = False
        self.video_path = ""
        self.cap_file = None
        self.cap_cam = None
        self.show_video_frame = True

        # New attributes for scoring
        self.ref_angles_sequence = []   # Store angle vectors of reference video
        self.current_frame_idx = 0
        self.scoring_active = False     # Whether scoring is enabled

        # Set up frames (left: video, right: webcam)
        self.left_frame = tk.Frame(self.root)
        self.left_frame.pack(side=tk.LEFT, padx=10)

        self.right_frame = tk.Frame(self.root)
        self.right_frame.pack(side=tk.RIGHT, padx=10)

        # Video File Window (Left)
        self.label_file = tk.Label(self.left_frame)
        self.label_file.pack()

        self.controls_file = tk.Frame(self.left_frame)
        self.controls_file.pack()

        tk.Button(self.controls_file, text="Open Video",
                  command=self.load_video).pack(side=tk.LEFT, padx=5)
        tk.Button(self.controls_file, text="Start Video",
                  command=self.start_video).pack(side=tk.LEFT, padx=5)
        tk.Button(self.controls_file, text="Stop Video",
                  command=self.stop_video).pack(side=tk.LEFT, padx=5)
        tk.Button(self.controls_file, text="Show/Hide Video",
                  command=self.toggle_video_display).pack(side=tk.LEFT, padx=5)

        # Webcam Window (Right)
        self.label_cam = tk.Label(self.right_frame)
        self.label_cam.pack()

        self.controls_cam = tk.Frame(self.right_frame)
        self.controls_cam.pack()

        tk.Button(self.controls_cam, text="Start Webcam",
                  command=self.start_cam).pack(side=tk.LEFT, padx=5)
        tk.Button(self.controls_cam, text="Stop Webcam",
                  command=self.stop_cam).pack(side=tk.LEFT, padx=5)

        # Scoring Controls and Display
        self.score_frame = tk.Frame(self.right_frame)
        self.score_frame.pack(pady=5)

        tk.Button(self.score_frame, text="Load Reference & Score",
                  command=self.start_scoring).pack(side=tk.LEFT, padx=5)

        self.score_label = tk.Label(self.score_frame, text="Score: --", font=("Arial", 14))
        self.score_label.pack(side=tk.LEFT, padx=10)

        self.feedback_label = tk.Label(self.right_frame, text="--", font=("Arial", 20))
        self.feedback_label.pack(pady=5)

    # Video Controls
    def load_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if path:
            self.video_path = path
            messagebox.showinfo("Video Selected", os.path.basename(path))

    def start_video(self):
        if not self.video_path:
            messagebox.showwarning("No Video", "Please select a video first.")
            return
        if not self.running_file:
            self.running_file = True
            threading.Thread(target=self.process_video_file, daemon=True).start()

    def stop_video(self):
        self.running_file = False
        if self.cap_file:
            self.cap_file.release()

    def toggle_video_display(self):
        self.show_video_frame = not self.show_video_frame

    # Webcam Controls
    def start_cam(self):
        if not self.running_cam:
            self.running_cam = True
            self.current_frame_idx = 0   # Reset frame index
            threading.Thread(target=self.process_webcam, daemon=True).start()

    def stop_cam(self):
        self.running_cam = False
        if self.cap_cam:
            self.cap_cam.release()
        self.current_frame_idx = 0

    # Scoring Controls
    def start_scoring(self):
        if not self.video_path:
            messagebox.showwarning("No Video", "Please open a reference video first.")
            return
        if not self.scoring_active:
            self.scoring_active = True
            # Preload reference angles in a separate thread
            threading.Thread(target=self.preload_reference_angles, daemon=True).start()

    def preload_reference_angles(self):
        """Extract angle vectors for every frame of the reference video."""
        self.ref_angles_sequence = []
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=0.3)
            if results and results[0].keypoints is not None:
                keypoints = extract_keypoints_from_result(results[0], frame.shape[1], frame.shape[0])
                # If all keypoints are None, treat as no person
                if keypoints is not None and any(p[0] is not None for p in keypoints):
                    angles = extract_angle_vector(keypoints)
                    self.ref_angles_sequence.append(angles)
                else:
                    self.ref_angles_sequence.append(np.zeros(len(ANGLE_TRIPLETS)))
            else:
                self.ref_angles_sequence.append(np.zeros(len(ANGLE_TRIPLETS)))
            frame_count += 1
        cap.release()
        print(f"Preloaded {len(self.ref_angles_sequence)} frames from reference.")
        # Update UI in main thread
        self.root.after(0, lambda: messagebox.showinfo("Ready", f"Loaded {len(self.ref_angles_sequence)} frames"))
        # Start webcam if not already running
        if not self.running_cam:
            self.start_cam()

    # Video Processing
    def process_video_file(self):
        self.cap_file = cv2.VideoCapture(self.video_path)
        while self.cap_file.isOpened() and self.running_file:
            ret, frame = self.cap_file.read()
            if not ret:
                break
            frame = self.process_pose(frame)
            self.update_label(self.label_file, frame)
        self.cap_file.release()

    def process_webcam(self):
        self.cap_cam = cv2.VideoCapture(0)
        self.current_frame_idx = 0
        while self.cap_cam.isOpened() and self.running_cam:
            ret, frame = self.cap_cam.read()
            if not ret:
                break

            # Process for display (draw skeleton)
            processed_frame = self.process_pose(frame)

            # Scoring logic
            if self.scoring_active and self.ref_angles_sequence:
                # Run inference to get keypoints for scoring
                results = model(frame, conf=0.3)
                if results and results[0].keypoints is not None:
                    keypoints = extract_keypoints_from_result(results[0], frame.shape[1], frame.shape[0])
                    if keypoints is not None and any(p[0] is not None for p in keypoints):
                        current_angles = extract_angle_vector(keypoints)
                        # Use modulo to loop reference if webcam runs longer
                        ref_idx = self.current_frame_idx % len(self.ref_angles_sequence)
                        ref_angles = self.ref_angles_sequence[ref_idx]
                        similarity = compute_angle_similarity(ref_angles, current_angles)
                        score = similarity  # 0-1
                        feedback, color = map_score_to_feedback(score)
                        # Update UI
                        self.root.after(0, self.update_score_display, score, feedback)
                        # Optionally draw feedback on frame
                        cv2.putText(processed_frame, f"{feedback} {score*100:.1f}%",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    else:
                        self.root.after(0, self.update_score_display, 0.0, "No person")
                else:
                    self.root.after(0, self.update_score_display, 0.0, "No person")
                self.current_frame_idx += 1

            # Update display
            self.update_label(self.label_cam, processed_frame)

        self.cap_cam.release()

    # Pose Drawing
    def process_pose(self, frame):
        results = model(frame, conf=0.3)
        height, width = frame.shape[:2]

        if self.show_video_frame:
            overlay = frame.copy()
        else:
            overlay = np.ones_like(frame) * 255  # white background

        for result in results:
            if result.keypoints is not None:
                keypoints_xyn = result.keypoints.xyn.cpu().numpy()
                for person_kpts in keypoints_xyn:
                    keypoints = [
                        (int(x * width), int(y * height)) for x, y in person_kpts
                    ]
                    # Draw skeleton lines
                    for pt1, pt2 in skeleton:
                        if pt1 < len(keypoints) and pt2 < len(keypoints):
                            x1, y1 = keypoints[pt1]
                            x2, y2 = keypoints[pt2]
                            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Draw keypoints
                    for x, y in keypoints:
                        cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)
        return overlay

    # Display Update
    def update_label(self, label, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = img.resize((640, 384))
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)

    def update_score_display(self, score, feedback):
        self.score_label.config(text=f"Score: {score*100:.1f}%")
        self.feedback_label.config(text=feedback)
        # Change color based on feedback
        if feedback == "Perfect!":
            self.feedback_label.config(fg="green")
        elif feedback == "Super!":
            self.feedback_label.config(fg="blue")
        elif feedback == "Good":
            self.feedback_label.config(fg="orange")
        else:
            self.feedback_label.config(fg="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = PoseApp(root)
    root.mainloop()