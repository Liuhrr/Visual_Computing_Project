"""Headless game session for the web-based Bonus Level frontend.

This module wraps the existing pose detection, scoring, and wall/freeze game
logic so it can be driven by a Flask backend instead of the Tkinter UI.
"""

from __future__ import annotations

import io
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from utils.pose_utils import (
    Pose,
    PoseSmoother,
    draw_pose,
    mirror_pose,
    pose_from_result,
    select_target_person,
    valid_keypoints,
)
from utils.reference import ReferenceSequence, analyze_reference_video
from utils.scoring import TemporalAligner, map_score_to_feedback
from wall_config import (
    DEFAULT_WALL_POSES_PATH,
    DEFAULT_WALL_SCHEDULE_PATH,
    WALL_SCORE_COMBO_CAP,
)
from wall_game import BonusEvent, FreezeGame, WallGame, build_wall_schedule
from extract_wall_poses import extract_wall_poses, load_wall_poses, save_wall_poses


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n-pose.pt"
CACHE_DIR = PROJECT_ROOT / "cache" / "reference"


@dataclass
class GameState:
    running: bool = False
    finished: bool = False
    elapsed: float = 0.0
    duration: float = 0.0
    dance_score: float = 0.0
    bonus_score: int = 0
    total_score: int = 0
    combo: int = 1
    grade: str = "—"
    feedback: str = "READY"
    feedback_color: tuple[int, int, int] = (80, 240, 80)
    wall_state: Optional[str] = None
    wall_match: float = 0.0
    freeze_active: bool = False
    freeze_message: str = ""
    camera_error: Optional[str] = None
    model_error: Optional[str] = None
    pose_visible: int = 0
    fps: float = 0.0
    latest_frame_jpeg: Optional[bytes] = None


class PoseWorker:
    """Background pose inference worker (mirrors LivePoseWorker from danceapp)."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._input: Optional[np.ndarray] = None
        self._output_lock = threading.Lock()
        self._output: Optional[tuple[int, Optional[Pose], Optional[str]]] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame_id: int, frame: np.ndarray) -> None:
        self._input = (frame_id, frame.copy())

    def latest(self) -> Optional[tuple[int, Optional[Pose], Optional[str]]]:
        with self._output_lock:
            out = self._output
            self._output = None
            return out

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.5)

    def _publish(self, payload: tuple[int, Optional[Pose], Optional[str]]) -> None:
        with self._output_lock:
            self._output = payload

    def _run(self) -> None:
        try:
            from ultralytics import YOLO
            model = YOLO(str(self.model_path))
        except Exception as exc:
            self._publish((-1, None, f"Unable to load pose model: {exc}"))
            return

        smoother = PoseSmoother(alpha=0.68, gap_decay=0.50)
        previous_center: Optional[np.ndarray] = None

        while not self._stop.is_set():
            item = self._input
            if item is None:
                time.sleep(0.01)
                continue
            self._input = None
            frame_id, frame = item
            try:
                result = model.predict(
                    source=frame,
                    conf=0.25,
                    imgsz=640,
                    max_det=6,
                    verbose=False,
                )[0]
                selection = select_target_person(
                    result,
                    frame.shape,
                    previous_center=previous_center,
                    conf_threshold=0.25,
                )
                pose = smoother.update(
                    pose_from_result(result, selection, conf_threshold=0.25)
                )
                if pose is not None and np.isfinite(pose.center).all():
                    previous_center = pose.center
                self._publish((frame_id, pose, None))
            except Exception as exc:
                self._publish((frame_id, None, f"Webcam inference failed: {exc}"))


class WebGameSession:
    """Runs the full Bonus Level game loop without Tkinter."""

    def __init__(self) -> None:
        self.state = GameState()
        self.video_path: Optional[Path] = None
        self.reference: Optional[ReferenceSequence] = None
        self.aligner: Optional[TemporalAligner] = None
        self.reference_capture: Optional[cv2.VideoCapture] = None
        self.camera_capture: Optional[cv2.VideoCapture] = None
        self.pose_worker: Optional[PoseWorker] = None
        self.wall_game: Optional[WallGame] = None
        self.freeze_game: Optional[FreezeGame] = None

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._start_time = 0.0
        self._live_frame_id = 0
        self._last_scored_frame_id = -1
        self._previous_score_pose: Optional[Pose] = None
        self._score_history: deque[float] = deque(maxlen=6000)
        self._feedback_counts = {
            "Perfect!": 0,
            "Super!": 0,
            "Good": 0,
            "Miss": 0,
        }
        self._pose_result_times: deque[float] = deque(maxlen=30)
        self._last_feedback = ""
        self._camera_failures = 0
        self._reference_frame_index = -1
        self._reference_frame: Optional[np.ndarray] = None

    def load_reference(self, video_path: Path) -> None:
        """Analyze a reference video and prepare wall/freeze assets."""
        self.video_path = video_path
        if not MODEL_PATH.is_file():
            self.state.model_error = f"Pose model not found: {MODEL_PATH}"
            return

        self.reference = analyze_reference_video(
            video_path,
            MODEL_PATH,
            CACHE_DIR,
            progress=lambda cur, total: None,
        )
        self.aligner = TemporalAligner(
            self.reference.poses,
            self.reference.timestamps,
            search_window_seconds=0.45,
        )
        self._ensure_wall_assets()

    def _ensure_wall_assets(self) -> None:
        if self.reference is None:
            return
        poses_path = DEFAULT_WALL_POSES_PATH
        schedule_path = DEFAULT_WALL_SCHEDULE_PATH
        if not poses_path.is_file():
            poses = extract_wall_poses(self.reference)
            save_wall_poses(poses, poses_path)
        if not schedule_path.is_file():
            poses = load_wall_poses(poses_path)
            build_wall_schedule(Path(self.reference.source_path), poses, schedule_path)

    def start(self) -> bool:
        if self.reference is None or self.aligner is None:
            self.state.camera_error = "No reference video loaded"
            return False
        if self.state.running:
            return True

        self.reference_capture = cv2.VideoCapture(self.reference.source_path)
        self.camera_capture = cv2.VideoCapture(0)

        if not self.reference_capture.isOpened():
            self.state.camera_error = "Unable to open reference video"
            self._release_captures()
            return False
        if not self.camera_capture.isOpened():
            self.state.camera_error = "Unable to open camera"
            self._release_captures()
            return False

        self.camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.pose_worker = PoseWorker(MODEL_PATH)

        # Initialize bonus games
        self.wall_game = None
        self.freeze_game = None
        if DEFAULT_WALL_POSES_PATH.is_file() and DEFAULT_WALL_SCHEDULE_PATH.is_file():
            try:
                self.wall_game = WallGame.from_files(
                    DEFAULT_WALL_POSES_PATH,
                    DEFAULT_WALL_SCHEDULE_PATH,
                    enabled=True,
                )
            except Exception:
                pass
        # Use FREEZE events that fit the reference duration so short clips
        # still get a FREEZE moment and long clips get two.
        ref_duration = self.reference.duration if self.reference else 0.0
        if ref_duration >= 18.0:
            freeze_events = [
                {"start": ref_duration * 0.30, "duration": 2.5},
                {"start": ref_duration * 0.65, "duration": 3.0},
            ]
        elif ref_duration >= 8.0:
            freeze_events = [
                {"start": ref_duration * 0.45, "duration": 2.5},
            ]
        else:
            from wall_config import FREEZE_EVENTS
            freeze_events = FREEZE_EVENTS
        self.freeze_game = FreezeGame(events=freeze_events, enabled=True)

        # Reset state
        self._start_time = time.perf_counter()
        self._live_frame_id = 0
        self._last_scored_frame_id = -1
        self._previous_score_pose = None
        self._score_history.clear()
        self._feedback_counts = {k: 0 for k in self._feedback_counts}
        self._pose_result_times.clear()
        self._last_feedback = ""
        self._camera_failures = 0
        self._reference_frame_index = -1
        self._reference_frame = None
        self.aligner.reset()

        self.state = GameState()
        self.state.running = True
        self.state.duration = self.reference.duration

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.pose_worker is not None:
            self.pose_worker.stop()
            self.pose_worker = None
        self._release_captures()
        with self._lock:
            self.state.running = False

    def _release_captures(self) -> None:
        for cap in (self.reference_capture, self.camera_capture):
            if cap is not None:
                cap.release()
        self.reference_capture = None
        self.camera_capture = None

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._tick()
            time.sleep(0.01)

    def _tick(self) -> None:
        if self.reference is None or self.aligner is None:
            return

        now = time.perf_counter()
        elapsed = now - self._start_time
        expected_index = min(
            int(elapsed * self.reference.fps), self.reference.frame_count - 1
        )

        if elapsed > self.reference.duration + 1.0 / self.reference.fps:
            self._finish()
            return

        # Reference frame
        ref_frame = self._read_reference_frame(expected_index)
        if ref_frame is None:
            self._finish()
            return

        ref_pose = self.reference.poses[expected_index]
        ref_display = draw_pose(
            ref_frame,
            ref_pose,
            line_color=(112, 124, 97),
            point_color=(236, 241, 238),
        )

        # Camera frame
        if self.camera_capture is None:
            return
        ok, camera_frame = self.camera_capture.read()
        if not ok or camera_frame is None:
            self._camera_failures += 1
            if self._camera_failures >= 5:
                with self._lock:
                    self.state.camera_error = "Camera disconnected"
                self._finish()
            return
        self._camera_failures = 0
        camera_frame = cv2.flip(camera_frame, 1)
        self._live_frame_id += 1

        # Pose inference
        live_pose: Optional[Pose] = None
        if self.pose_worker is not None:
            self.pose_worker.submit(self._live_frame_id, camera_frame)
            output = self.pose_worker.latest()
            if output is not None:
                frame_id, pose, error = output
                if error:
                    with self._lock:
                        self.state.model_error = error
                    self._finish()
                    return
                live_pose = pose
                self._pose_result_times.append(now)

        # Dance scoring
        scoreable = self._is_scoreable_pose(live_pose)
        if scoreable and live_pose is not None:
            if self._live_frame_id != self._last_scored_frame_id:
                score_pose = mirror_pose(live_pose)
                alignment = self.aligner.align(
                    score_pose,
                    elapsed,
                    previous_user_pose=self._previous_score_pose,
                )
                self._previous_score_pose = score_pose
                self._last_scored_frame_id = self._live_frame_id
                self._update_score(alignment.breakdown)

        # Render player view
        camera_display = draw_pose(
            camera_frame,
            live_pose,
            line_color=(138, 117, 97),
            point_color=(235, 231, 225),
        )

        # Bonus games
        bonus_events: list[BonusEvent] = []
        if self.wall_game is not None:
            camera_display, events = self.wall_game.update(
                elapsed, live_pose, camera_display
            )
            bonus_events.extend(events)

        if self.freeze_game is not None:
            camera_display, event = self.freeze_game.update(
                elapsed, live_pose, camera_display
            )
            if event is not None:
                bonus_events.append(event)

        for event in bonus_events:
            self.state.bonus_score += event.score
            if event.combo > 0:
                self.state.combo = min(event.combo, WALL_SCORE_COMBO_CAP)
            else:
                self.state.combo = 1

        # Compose main display: reference left, player right
        display = np.zeros((max(ref_display.shape[0], camera_display.shape[0]),
                            ref_display.shape[1] + camera_display.shape[1], 3), dtype=np.uint8)
        h1, w1 = ref_display.shape[:2]
        h2, w2 = camera_display.shape[:2]
        display[:h1, :w1] = ref_display
        display[:h2, w1:w1+w2] = camera_display

        # Encode to JPEG
        ok, jpeg = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            with self._lock:
                self.state.latest_frame_jpeg = jpeg.tobytes()
                self.state.elapsed = elapsed
                self.state.fps = self._pose_fps()
                self.state.pose_visible = (
                    int(np.count_nonzero(valid_keypoints(live_pose)[5:17]))
                    if live_pose is not None else 0
                )
                if self.wall_game is not None and self.wall_game.active_wall is not None:
                    self.state.wall_state = self.wall_game.active_wall.state.value
                    self.state.wall_match = self.wall_game.active_wall.similarity
                else:
                    self.state.wall_state = None
                    self.state.wall_match = 0.0
                self.state.freeze_active = (
                    self.freeze_game is not None and self.freeze_game.active_index is not None
                )

    def _read_reference_frame(self, target_index: int) -> Optional[np.ndarray]:
        cap = self.reference_capture
        if cap is None:
            return None
        if target_index < self._reference_frame_index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
            self._reference_frame_index = target_index - 1
        while self._reference_frame_index < target_index:
            ok, frame = cap.read()
            if not ok:
                return None
            self._reference_frame = frame
            self._reference_frame_index += 1
        return self._reference_frame.copy() if self._reference_frame is not None else None

    @staticmethod
    def _is_scoreable_pose(pose: Optional[Pose]) -> bool:
        if pose is None:
            return False
        body = valid_keypoints(pose)[5:17]
        torso = valid_keypoints(pose)[5:13]
        return bool(np.count_nonzero(body) >= 4 and np.count_nonzero(torso) >= 2)

    def _pose_fps(self) -> float:
        if len(self._pose_result_times) < 2:
            return 0.0
        duration = self._pose_result_times[-1] - self._pose_result_times[0]
        if duration <= 0:
            return 0.0
        return (len(self._pose_result_times) - 1) / duration

    def _update_score(self, breakdown) -> None:
        self._score_history.append(breakdown.total)
        average = float(np.mean(self._score_history)) * 100.0
        feedback, bgr = map_score_to_feedback(breakdown.total)
        self._feedback_counts[feedback] += 1

        with self._lock:
            self.state.dance_score = average
            self.state.feedback = feedback.upper().rstrip("!")
            self.state.feedback_color = bgr

        self._last_feedback = feedback

    def _finish(self) -> None:
        with self._lock:
            self.state.running = False
            self.state.finished = True
            dance_score_int = int(round(self.state.dance_score * 10))
            total = dance_score_int + self.state.bonus_score
            self.state.total_score = total
            self.state.grade = self._compute_grade(total)
        self._stop_event.set()

    @staticmethod
    def _compute_grade(total: int) -> str:
        if total >= 900:
            return "S"
        if total >= 750:
            return "A"
        if total >= 600:
            return "B"
        return "C"

    def get_state_snapshot(self) -> GameState:
        with self._lock:
            return GameState(
                running=self.state.running,
                finished=self.state.finished,
                elapsed=self.state.elapsed,
                duration=self.state.duration,
                dance_score=self.state.dance_score,
                bonus_score=self.state.bonus_score,
                total_score=self.state.total_score,
                combo=self.state.combo,
                grade=self.state.grade,
                feedback=self.state.feedback,
                feedback_color=self.state.feedback_color,
                wall_state=self.state.wall_state,
                wall_match=self.state.wall_match,
                freeze_active=self.state.freeze_active,
                freeze_message=self.state.freeze_message,
                camera_error=self.state.camera_error,
                model_error=self.state.model_error,
                pose_visible=self.state.pose_visible,
                fps=self.state.fps,
            )
