"""Pose or Die: a webcam silhouette-matching survival game.

The game reuses the YOLOv8-pose pipeline from the supplied visual computing
project.  A random white silhouette is shown on a black stage.  The player has
a few seconds to copy it and hold the pose; otherwise the run ends.
"""

from __future__ import annotations

import os
import queue
import random
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

from utils.pose_utils import (
    Pose,
    PoseSmoother,
    draw_pose,
    mirror_pose,
    pose_from_result,
    select_target_person,
    valid_keypoints,
)
from utils.scoring import compute_pose_score


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n-pose.pt"
CACHE_ROOT = PROJECT_ROOT / "cache"
MPL_CACHE = CACHE_ROOT / "matplotlib"
YOLO_CACHE = CACHE_ROOT / "ultralytics"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
YOLO_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CACHE))

BODY_INDICES = np.arange(5, 17)
TORSO_INDICES = np.array([5, 6, 11, 12])
MATCH_THRESHOLD = 0.70
HOLD_SECONDS = 0.55
PREP_SECONDS = 2.5
STARTING_ROUND_SECONDS = 5.5
MINIMUM_ROUND_SECONDS = 3.0
CAMERA_PREVIEW_WIDTH = 480
CAMERA_PREVIEW_HEIGHT = 360


class GameState(str, Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    PLAYING = "playing"
    SUCCESS = "success"
    DEAD = "dead"


@dataclass(frozen=True)
class ActionPose:
    name: str
    hint: str
    pose: Pose


def _pose_from_points(points: dict[int, tuple[float, float]]) -> Pose:
    """Create a normalized COCO pose from the body points in ``points``."""

    xy = np.full((17, 2), np.nan, dtype=np.float32)
    confidence = np.zeros(17, dtype=np.float32)
    for index, point in points.items():
        xy[index] = point
        confidence[index] = 1.0
    return Pose(xy=xy, confidence=confidence)


def _base_points() -> dict[int, tuple[float, float]]:
    return {
        0: (0.50, 0.13),
        5: (0.41, 0.28),
        6: (0.59, 0.28),
        7: (0.36, 0.42),
        8: (0.64, 0.42),
        9: (0.38, 0.54),
        10: (0.62, 0.54),
        11: (0.44, 0.54),
        12: (0.56, 0.54),
        13: (0.43, 0.72),
        14: (0.57, 0.72),
        15: (0.42, 0.93),
        16: (0.58, 0.93),
    }


def _action(
    name: str,
    hint: str,
    overrides: dict[int, tuple[float, float]],
) -> ActionPose:
    points = _base_points()
    points.update(overrides)
    return ActionPose(name=name, hint=hint, pose=_pose_from_points(points))


ACTION_POSES: tuple[ActionPose, ...] = (
    _action(
        "大字星",
        "双臂张开，双脚分开",
        {
            7: (0.27, 0.31),
            8: (0.73, 0.31),
            9: (0.11, 0.20),
            10: (0.89, 0.20),
            13: (0.34, 0.72),
            14: (0.66, 0.72),
            15: (0.20, 0.93),
            16: (0.80, 0.93),
        },
    ),
    _action(
        "双手举高",
        "手臂伸向头顶",
        {
            7: (0.39, 0.16),
            8: (0.61, 0.16),
            9: (0.37, 0.02),
            10: (0.63, 0.02),
        },
    ),
    _action(
        "飞机",
        "双臂水平伸直",
        {
            7: (0.27, 0.28),
            8: (0.73, 0.28),
            9: (0.08, 0.28),
            10: (0.92, 0.28),
            13: (0.40, 0.72),
            15: (0.36, 0.93),
            14: (0.62, 0.70),
            16: (0.72, 0.62),
        },
    ),
    _action(
        "深蹲",
        "屈膝下蹲，手臂向前",
        {
            5: (0.42, 0.31),
            6: (0.58, 0.31),
            7: (0.38, 0.40),
            8: (0.62, 0.40),
            9: (0.30, 0.44),
            10: (0.70, 0.44),
            11: (0.40, 0.56),
            12: (0.60, 0.56),
            13: (0.27, 0.70),
            14: (0.73, 0.70),
            15: (0.21, 0.88),
            16: (0.79, 0.88),
        },
    ),
    _action(
        "迪斯科",
        "一只手举高，另一只手叉腰",
        {
            7: (0.34, 0.16),
            9: (0.27, 0.03),
            8: (0.72, 0.40),
            10: (0.57, 0.53),
            13: (0.36, 0.72),
            14: (0.64, 0.72),
            15: (0.27, 0.93),
            16: (0.73, 0.93),
        },
    ),
    _action(
        "拳击防守",
        "双拳举到脸旁",
        {
            7: (0.34, 0.37),
            8: (0.66, 0.37),
            9: (0.43, 0.18),
            10: (0.57, 0.18),
            13: (0.38, 0.72),
            14: (0.62, 0.72),
            15: (0.29, 0.93),
            16: (0.71, 0.93),
        },
    ),
    _action(
        "单脚平衡",
        "双臂张开，抬起一条腿",
        {
            7: (0.27, 0.28),
            8: (0.73, 0.28),
            9: (0.08, 0.28),
            10: (0.92, 0.28),
            13: (0.43, 0.72),
            15: (0.42, 0.93),
            14: (0.66, 0.66),
            16: (0.54, 0.61),
        },
    ),
)


def choose_next_action(
    previous_index: Optional[int],
    rng: random.Random,
) -> int:
    """Return a random action index without immediately repeating a pose."""

    candidates = list(range(len(ACTION_POSES)))
    if previous_index in candidates and len(candidates) > 1:
        candidates.remove(previous_index)
    return rng.choice(candidates)


def pose_is_visible(pose: Optional[Pose]) -> bool:
    """Require a reliable torso and most limbs before accepting a match."""

    if pose is None:
        return False
    visible = valid_keypoints(pose)
    torso_ok = int(np.count_nonzero(visible[TORSO_INDICES])) >= 3
    body_ok = int(np.count_nonzero(visible[BODY_INDICES])) >= 8
    return torso_ok and body_ok


def best_pose_score(reference: Pose, player: Optional[Pose]) -> float:
    """Score both anatomical directions so mirrored webcams feel natural."""

    if not pose_is_visible(player):
        return 0.0
    assert player is not None
    direct = compute_pose_score(reference, player).total
    mirrored = compute_pose_score(reference, mirror_pose(player)).total
    return float(max(direct, mirrored))


def fit_camera_frame(
    frame: np.ndarray,
    width: int = CAMERA_PREVIEW_WIDTH,
    height: int = CAMERA_PREVIEW_HEIGHT,
) -> np.ndarray:
    """Fit a camera frame into a fixed viewport without cropping or stretching."""

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected a BGR camera frame with three channels")
    source_height, source_width = frame.shape[:2]
    if source_width <= 0 or source_height <= 0 or width <= 0 or height <= 0:
        raise ValueError("Camera frame and viewport dimensions must be positive")

    scale = min(width / source_width, height / source_height)
    fitted_width = max(1, int(round(source_width * scale)))
    fitted_height = max(1, int(round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    fitted = cv2.resize(
        frame,
        (fitted_width, fitted_height),
        interpolation=interpolation,
    )
    canvas = np.full((height, width, 3), 10, dtype=frame.dtype)
    x0 = (width - fitted_width) // 2
    y0 = (height - fitted_height) // 2
    canvas[y0:y0 + fitted_height, x0:x0 + fitted_width] = fitted
    return canvas


class LivePoseWorker:
    """Run YOLO inference away from Tk's UI thread."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._inputs: queue.Queue[Optional[tuple[int, np.ndarray]]] = queue.Queue(
            maxsize=1
        )
        self._outputs: queue.Queue[tuple[int, Optional[Pose], Optional[str]]] = (
            queue.Queue(maxsize=1)
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="silhouette-pose-worker",
            daemon=True,
        )
        self._thread.start()

    def submit(self, frame_id: int, frame: np.ndarray) -> None:
        item = (frame_id, frame.copy())
        try:
            self._inputs.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            self._inputs.get_nowait()
        except queue.Empty:
            pass
        try:
            self._inputs.put_nowait(item)
        except queue.Full:
            pass

    def latest(self) -> Optional[tuple[int, Optional[Pose], Optional[str]]]:
        newest = None
        while True:
            try:
                newest = self._outputs.get_nowait()
            except queue.Empty:
                return newest

    def stop(self) -> None:
        self._stop.set()
        try:
            self._inputs.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    def _publish(
        self,
        payload: tuple[int, Optional[Pose], Optional[str]],
    ) -> None:
        try:
            self._outputs.put_nowait(payload)
            return
        except queue.Full:
            pass
        try:
            self._outputs.get_nowait()
        except queue.Empty:
            pass
        try:
            self._outputs.put_nowait(payload)
        except queue.Full:
            pass

    def _run(self) -> None:
        try:
            from ultralytics import YOLO

            model = YOLO(str(self.model_path))
        except Exception as exc:
            self._publish((-1, None, f"姿态模型加载失败：{exc}"))
            return

        smoother = PoseSmoother(alpha=0.68, gap_decay=0.50)
        previous_center: Optional[np.ndarray] = None
        while not self._stop.is_set():
            try:
                item = self._inputs.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            frame_id, frame = item
            try:
                result = model.predict(
                    source=frame,
                    conf=0.25,
                    imgsz=640,
                    max_det=4,
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
                self._publish((frame_id, None, f"姿态识别失败：{exc}"))


class SilhouetteGame:
    TICK_MS = 30

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("POSE OR DIE · 姿势生存")
        self.root.geometry("1240x780")
        self.root.minsize(1000, 650)
        self.root.configure(bg="#000000")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<space>", lambda _event: self.start_game())
        self.root.bind("<Escape>", lambda _event: self.close())

        self.state = GameState.IDLE
        self.rng = random.Random()
        self.action_index: Optional[int] = None
        self.round_number = 0
        self.score = 0
        self.high_score = 0
        self.deadline = 0.0
        self.prep_deadline = 0.0
        self.hold_started: Optional[float] = None
        self.success_until = 0.0
        self.latest_pose: Optional[Pose] = None
        self.latest_pose_time = 0.0
        self.latest_match = 0.0
        self.frame_id = 0
        self.last_result_frame = -1
        self.camera_failures = 0
        self.camera_capture: Optional[cv2.VideoCapture] = None
        self.worker: Optional[LivePoseWorker] = None
        self.after_id: Optional[str] = None
        self.camera_photo: Optional[ImageTk.PhotoImage] = None

        self.timer_text = tk.StringVar(value="--")
        self.round_text = tk.StringVar(value="回合 0")
        self.score_text = tk.StringVar(value="得分 0")
        self.action_text = tk.StringVar(value="准备好了吗？")
        self.hint_text = tk.StringVar(value="点击开始，然后在倒计时结束前模仿白色人形")
        self.status_text = tk.StringVar(value="正在连接摄像头与姿态模型…")
        self.match_text = tk.StringVar(value="匹配度 --")
        self.button_text = tk.StringVar(value="开始游戏  SPACE")

        self._build_ui()
        self._open_camera()
        self.worker = LivePoseWorker(MODEL_PATH)
        self._render_silhouette()
        self.after_id = self.root.after(self.TICK_MS, self._tick)

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, bg="#000000")
        header.pack(fill=tk.X, padx=34, pady=(24, 14))

        title_block = tk.Frame(header, bg="#000000")
        title_block.pack(side=tk.LEFT)
        tk.Label(
            title_block,
            text="POSE OR DIE",
            bg="#000000",
            fg="#FFFFFF",
            font=("Arial", 25, "bold"),
        ).pack(anchor=tk.W)
        tk.Label(
            title_block,
            text="姿势生存",
            bg="#000000",
            fg="#777777",
            font=("Microsoft YaHei UI", 11),
        ).pack(anchor=tk.W, pady=(2, 0))

        metrics = tk.Frame(header, bg="#000000")
        metrics.pack(side=tk.RIGHT)
        for variable in (self.round_text, self.score_text, self.timer_text):
            tk.Label(
                metrics,
                textvariable=variable,
                bg="#000000",
                fg="#FFFFFF",
                font=("Microsoft YaHei UI", 14, "bold"),
                padx=18,
            ).pack(side=tk.LEFT)

        content = tk.Frame(self.root, bg="#000000")
        content.pack(fill=tk.BOTH, expand=True, padx=34)
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(
            1,
            weight=0,
            minsize=CAMERA_PREVIEW_WIDTH,
        )

        stage = tk.Frame(
            content,
            bg="#000000",
            highlightbackground="#2A2A2A",
            highlightthickness=1,
        )
        stage.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        stage.grid_rowconfigure(1, weight=1)
        stage.grid_columnconfigure(0, weight=1)

        stage_top = tk.Frame(stage, bg="#000000")
        stage_top.grid(row=0, column=0, sticky="ew", padx=22, pady=(18, 6))
        tk.Label(
            stage_top,
            textvariable=self.action_text,
            bg="#000000",
            fg="#FFFFFF",
            font=("Microsoft YaHei UI", 19, "bold"),
        ).pack(side=tk.LEFT)
        tk.Label(
            stage_top,
            textvariable=self.match_text,
            bg="#000000",
            fg="#8BFF9B",
            font=("Microsoft YaHei UI", 12, "bold"),
        ).pack(side=tk.RIGHT)

        self.pose_canvas = tk.Canvas(
            stage,
            bg="#000000",
            bd=0,
            highlightthickness=0,
        )
        self.pose_canvas.grid(row=1, column=0, sticky="nsew")
        self.pose_canvas.bind(
            "<Configure>",
            lambda _event: self._render_silhouette(),
        )
        tk.Label(
            stage,
            textvariable=self.hint_text,
            bg="#000000",
            fg="#AAAAAA",
            font=("Microsoft YaHei UI", 11),
            pady=14,
        ).grid(row=2, column=0, sticky="ew")

        camera_card = tk.Frame(
            content,
            bg="#0A0A0A",
            width=CAMERA_PREVIEW_WIDTH,
            highlightbackground="#2A2A2A",
            highlightthickness=1,
        )
        camera_card.grid(row=0, column=1, sticky="n", padx=(10, 0))
        camera_card.grid_rowconfigure(1, weight=0)
        camera_card.grid_columnconfigure(0, weight=1)

        camera_header = tk.Frame(camera_card, bg="#0A0A0A")
        camera_header.grid(row=0, column=0, sticky="ew", padx=18, pady=16)
        tk.Label(
            camera_header,
            text="摄像头画面",
            bg="#0A0A0A",
            fg="#FFFFFF",
            font=("Microsoft YaHei UI", 13, "bold"),
        ).pack(side=tk.LEFT)
        tk.Label(
            camera_header,
            text="LIVE",
            bg="#0A0A0A",
            fg="#FF5252",
            font=("Arial", 10, "bold"),
        ).pack(side=tk.RIGHT)

        camera_viewport = tk.Frame(
            camera_card,
            bg="#0A0A0A",
            width=CAMERA_PREVIEW_WIDTH,
            height=CAMERA_PREVIEW_HEIGHT,
        )
        camera_viewport.grid(row=1, column=0)
        camera_viewport.grid_propagate(False)

        self.camera_label = tk.Label(
            camera_viewport,
            bg="#0A0A0A",
            fg="#888888",
            text="正在启动摄像头…",
            font=("Microsoft YaHei UI", 12),
            compound=tk.CENTER,
        )
        self.camera_label.place(
            x=0,
            y=0,
            width=CAMERA_PREVIEW_WIDTH,
            height=CAMERA_PREVIEW_HEIGHT,
        )

        status_bar = tk.Frame(camera_card, bg="#0A0A0A")
        status_bar.grid(row=2, column=0, sticky="ew", padx=18, pady=16)
        tk.Label(
            status_bar,
            textvariable=self.status_text,
            bg="#0A0A0A",
            fg="#AAAAAA",
            font=("Microsoft YaHei UI", 10),
            wraplength=390,
            justify=tk.LEFT,
        ).pack(anchor=tk.W)

        footer = tk.Frame(self.root, bg="#000000")
        footer.pack(fill=tk.X, padx=34, pady=(16, 24))
        tk.Label(
            footer,
            text="站远一些，让肩膀、手腕、髋部、膝盖和脚踝都进入画面",
            bg="#000000",
            fg="#777777",
            font=("Microsoft YaHei UI", 10),
        ).pack(side=tk.LEFT)

        self.start_button = tk.Button(
            footer,
            textvariable=self.button_text,
            command=self.start_game,
            bg="#FFFFFF",
            fg="#000000",
            activebackground="#DDDDDD",
            activeforeground="#000000",
            relief=tk.FLAT,
            bd=0,
            padx=28,
            pady=12,
            cursor="hand2",
            font=("Microsoft YaHei UI", 11, "bold"),
        )
        self.start_button.pack(side=tk.RIGHT)

    def _open_camera(self) -> None:
        if self.camera_capture is not None:
            self.camera_capture.release()
        if os.name == "nt":
            capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(0)
        else:
            capture = cv2.VideoCapture(0)

        if not capture.isOpened():
            capture.release()
            self.camera_capture = None
            self.status_text.set("无法打开摄像头。请检查系统权限或是否被其他应用占用。")
            return

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera_capture = capture
        self.status_text.set("摄像头已连接，姿态模型正在加载…")

    def start_game(self) -> None:
        if self.state in (GameState.PREPARING, GameState.PLAYING, GameState.SUCCESS):
            return
        if self.camera_capture is None:
            self._open_camera()
        if self.camera_capture is None:
            self.status_text.set("没有可用摄像头，暂时无法开始。")
            return

        self.state = GameState.PREPARING
        self.round_number = 0
        self.score = 0
        self.action_index = None
        self.latest_match = 0.0
        self.hold_started = None
        self.prep_deadline = time.monotonic() + PREP_SECONDS
        self.round_text.set("回合 0")
        self.score_text.set("得分 0")
        self.action_text.set("进入画面")
        self.hint_text.set("全身尽量完整地出现在摄像头中")
        self.button_text.set("游戏进行中")
        self.start_button.configure(state=tk.DISABLED, bg="#333333", fg="#888888")
        self._render_silhouette()

    def _start_round(self) -> None:
        self.action_index = choose_next_action(self.action_index, self.rng)
        self.round_number += 1
        duration = max(
            MINIMUM_ROUND_SECONDS,
            STARTING_ROUND_SECONDS - 0.22 * (self.round_number - 1),
        )
        self.deadline = time.monotonic() + duration
        self.state = GameState.PLAYING
        self.hold_started = None
        self.latest_match = 0.0
        action = ACTION_POSES[self.action_index]
        self.action_text.set(action.name)
        self.hint_text.set(action.hint)
        self.round_text.set(f"回合 {self.round_number}")
        self.match_text.set("匹配度 --")
        self.status_text.set("模仿左侧人形，并保持动作约半秒")
        self._render_silhouette()

    def _mark_success(self, now: float) -> None:
        self.state = GameState.SUCCESS
        self.score += 1
        self.high_score = max(self.high_score, self.score)
        self.success_until = now + 0.75
        self.hold_started = None
        self.score_text.set(f"得分 {self.score}")
        self.action_text.set("MATCHED")
        self.hint_text.set("动作匹配，准备下一回合")
        self.status_text.set("成功")
        self._render_silhouette()

    def _die(self) -> None:
        self.state = GameState.DEAD
        self.hold_started = None
        self.timer_text.set("0.0")
        self.action_text.set("YOU DIED")
        self.hint_text.set(
            f"本次完成 {self.score} 个动作 · 最高记录 {self.high_score}"
        )
        self.status_text.set("时间耗尽。按空格或点击按钮重新开始。")
        self.button_text.set("再试一次  SPACE")
        self.start_button.configure(
            state=tk.NORMAL,
            bg="#FFFFFF",
            fg="#000000",
        )
        self._render_silhouette()

    def _update_game(self, now: float) -> None:
        if self.state == GameState.IDLE:
            self.timer_text.set("--")
            return

        if self.state == GameState.PREPARING:
            remaining = max(0.0, self.prep_deadline - now)
            self.timer_text.set(str(max(1, int(np.ceil(remaining)))))
            if remaining <= 0.0:
                self._start_round()
            return

        if self.state == GameState.SUCCESS:
            self.timer_text.set("✓")
            if now >= self.success_until:
                self._start_round()
            return

        if self.state != GameState.PLAYING:
            return

        remaining = max(0.0, self.deadline - now)
        self.timer_text.set(f"{remaining:.1f}")
        if remaining <= 0.0:
            self._die()
            return

        if self.action_index is None:
            return
        if now - self.latest_pose_time > 0.8:
            self.latest_match = 0.0
        else:
            self.latest_match = best_pose_score(
                ACTION_POSES[self.action_index].pose,
                self.latest_pose,
            )
        self.match_text.set(f"匹配度 {self.latest_match * 100:.0f}%")

        if self.latest_match >= MATCH_THRESHOLD:
            if self.hold_started is None:
                self.hold_started = now
            held = now - self.hold_started
            self.status_text.set(f"保持动作… {min(held / HOLD_SECONDS, 1.0) * 100:.0f}%")
            if held >= HOLD_SECONDS:
                self._mark_success(now)
        else:
            self.hold_started = None
            if pose_is_visible(self.latest_pose):
                self.status_text.set("继续调整动作，让轮廓更接近")
            else:
                self.status_text.set("未看清全身，请后退并保持光线充足")

    def _tick(self) -> None:
        now = time.monotonic()
        frame: Optional[np.ndarray] = None
        if self.camera_capture is not None:
            ok, raw = self.camera_capture.read()
            if ok and raw is not None:
                self.camera_failures = 0
                frame = cv2.flip(raw, 1)
                self.frame_id += 1
                if self.worker is not None:
                    self.worker.submit(self.frame_id, frame)
            else:
                self.camera_failures += 1
                if self.camera_failures >= 10:
                    self.status_text.set("摄像头画面中断，请关闭占用摄像头的其他应用。")

        result = self.worker.latest() if self.worker is not None else None
        if result is not None:
            result_frame_id, pose, error = result
            if error:
                self.status_text.set(error)
            elif result_frame_id >= 0 and result_frame_id != self.last_result_frame:
                first_result = self.last_result_frame < 0
                self.last_result_frame = result_frame_id
                self.latest_pose = pose
                self.latest_pose_time = now
                if first_result and self.state == GameState.IDLE:
                    self.status_text.set("姿态模型已就绪。点击开始或按空格键。")

        if frame is not None:
            display = draw_pose(
                frame,
                self.latest_pose,
                line_color=(110, 255, 130),
                point_color=(255, 255, 255),
            )
            self._show_camera_frame(display)

        self._update_game(now)
        self.after_id = self.root.after(self.TICK_MS, self._tick)

    def _show_camera_frame(self, frame: np.ndarray) -> None:
        fixed_frame = fit_camera_frame(frame)
        rgb = cv2.cvtColor(fixed_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self.camera_photo = ImageTk.PhotoImage(image)
        self.camera_label.configure(image=self.camera_photo, text="")

    def _pose_to_canvas(
        self,
        pose: Pose,
        width: int,
        height: int,
    ) -> np.ndarray:
        drawing_height = max(1.0, height * 0.86)
        drawing_width = min(width * 0.84, drawing_height * 0.78)
        x0 = (width - drawing_width) / 2.0
        y0 = (height - drawing_height) / 2.0
        points = pose.xy.copy()
        points[:, 0] = x0 + points[:, 0] * drawing_width
        points[:, 1] = y0 + points[:, 1] * drawing_height
        return points

    def _render_silhouette(self) -> None:
        if not hasattr(self, "pose_canvas"):
            return
        canvas = self.pose_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 400)
        height = max(canvas.winfo_height(), 420)

        if self.state == GameState.DEAD:
            canvas.create_text(
                width / 2,
                height / 2 - 18,
                text="×",
                fill="#FF3B3B",
                font=("Arial", 110, "bold"),
            )
            canvas.create_text(
                width / 2,
                height / 2 + 70,
                text="TIME OUT",
                fill="#FFFFFF",
                font=("Arial", 18, "bold"),
            )
            return

        if self.state == GameState.PREPARING:
            canvas.create_text(
                width / 2,
                height / 2,
                text="GET READY",
                fill="#FFFFFF",
                font=("Arial", 34, "bold"),
            )
            return

        action = (
            ACTION_POSES[self.action_index]
            if self.action_index is not None
            else ACTION_POSES[0]
        )
        points = self._pose_to_canvas(action.pose, width, height)
        scale = min(width, height)
        limb_width = max(16, int(scale * 0.055))
        joint_radius = limb_width * 0.47
        color = "#8BFF9B" if self.state == GameState.SUCCESS else "#FFFFFF"

        # Limbs first, then the torso and head, forming one solid blank figure.
        for start, end in ((5, 7), (7, 9), (6, 8), (8, 10),
                           (11, 13), (13, 15), (12, 14), (14, 16)):
            p1 = points[start]
            p2 = points[end]
            canvas.create_line(
                p1[0],
                p1[1],
                p2[0],
                p2[1],
                fill=color,
                width=limb_width,
                capstyle=tk.ROUND,
                joinstyle=tk.ROUND,
            )
        torso = [
            *points[5],
            *points[6],
            *points[12],
            *points[11],
        ]
        canvas.create_polygon(torso, fill=color, outline=color)
        for index in (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16):
            x, y = points[index]
            canvas.create_oval(
                x - joint_radius,
                y - joint_radius,
                x + joint_radius,
                y + joint_radius,
                fill=color,
                outline=color,
            )

        shoulder_center = (points[5] + points[6]) / 2.0
        head_center = points[0]
        neck_width = max(12, int(limb_width * 0.72))
        canvas.create_line(
            shoulder_center[0],
            shoulder_center[1],
            head_center[0],
            head_center[1] + limb_width * 0.85,
            fill=color,
            width=neck_width,
            capstyle=tk.ROUND,
        )
        head_radius = max(limb_width * 0.95, abs(points[6, 0] - points[5, 0]) * 0.34)
        canvas.create_oval(
            head_center[0] - head_radius,
            head_center[1] - head_radius,
            head_center[0] + head_radius,
            head_center[1] + head_radius,
            fill=color,
            outline=color,
        )

    def close(self) -> None:
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        if self.camera_capture is not None:
            self.camera_capture.release()
            self.camera_capture = None
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    SilhouetteGame(root)
    root.mainloop()


if __name__ == "__main__":
    main()
