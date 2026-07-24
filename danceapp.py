"""Restrained desktop UI for the NUS Visual Computing Just Dance project."""

from __future__ import annotations

import os
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk

from utils.pose_utils import (
    Pose,
    PoseSmoother,
    draw_pose,
    mirror_pose,
    pose_from_result,
    select_target_person,
    valid_keypoints,
)
from utils.reference import (
    ReferenceAnalysisCancelled,
    ReferenceSequence,
    analyze_reference_video,
)
from utils.scoring import ScoreBreakdown, TemporalAligner, map_score_to_feedback, WindowScorer


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n-pose.pt"
CACHE_DIR = PROJECT_ROOT / "cache" / "reference"
os.environ.setdefault(
    "MPLCONFIGDIR", str(PROJECT_ROOT / "cache" / "matplotlib")
)


class Palette:
    CANVAS = "#F5F5F2"
    SURFACE = "#FFFFFF"
    TEXT = "#202124"
    MUTED = "#74777C"
    FAINT = "#A5A8A4"
    BORDER = "#E4E5E2"
    ACCENT = "#617C70"
    ACCENT_HOVER = "#526C61"
    ACCENT_SOFT = "#E8EEEA"
    BLUE = "#61758A"
    AMBER = "#A67C42"
    AMBER_SOFT = "#F3EBDD"
    RED = "#9A625D"
    RED_SOFT = "#F3E7E5"
    VIDEO = "#1E211F"
    VIDEO_MUTED = "#C5C9C5"
    WHITE = "#FFFFFF"


FEEDBACK_COLORS = {
    "Perfect!": "#4E7A62",
    "Super!": Palette.BLUE,
    "Good": Palette.AMBER,
    "Miss": Palette.RED,
}


class UIState(str, Enum):
    EMPTY = "empty"
    VIDEO_LOADED = "video_loaded"
    ANALYZING = "analyzing"
    READY = "ready"
    RUNNING = "running"
    POSE_LOST = "pose_lost"
    CAMERA_ERROR = "camera_error"
    FINISHED = "finished"


def _rounded_polygon(
    canvas: tk.Canvas,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    radius: float,
    **kwargs,
) -> int:
    radius = max(0.0, min(radius, (x2 - x1) / 2.0, (y2 - y1) / 2.0))
    points = [
        x1 + radius,
        y1,
        x2 - radius,
        y1,
        x2,
        y1,
        x2,
        y1 + radius,
        x2,
        y2 - radius,
        x2,
        y2,
        x2 - radius,
        y2,
        x1 + radius,
        y2,
        x1,
        y2,
        x1,
        y2 - radius,
        x1,
        y1 + radius,
        x1,
        y1,
    ]
    return canvas.create_polygon(
        points, smooth=True, splinesteps=24, **kwargs
    )


class RoundedPanel(tk.Canvas):
    """Canvas-backed card with a real Tk frame for child widgets."""

    def __init__(
        self,
        parent,
        *,
        fill: str = Palette.SURFACE,
        outline: str = Palette.BORDER,
        radius: int = 10,
        padding: int = 12,
        **kwargs,
    ) -> None:
        parent_bg = parent.cget("bg")
        super().__init__(
            parent,
            bg=parent_bg,
            highlightthickness=0,
            bd=0,
            **kwargs,
        )
        self.fill = fill
        self.outline = outline
        self.radius = radius
        self.padding = padding
        self.content = tk.Frame(self, bg=fill)
        self._window = self.create_window(
            padding,
            padding,
            anchor=tk.NW,
            window=self.content,
        )
        self.bind("<Configure>", self._redraw)

    def _redraw(self, _event=None) -> None:
        width = max(self.winfo_width(), 2)
        height = max(self.winfo_height(), 2)
        self.delete("panel")
        _rounded_polygon(
            self,
            1,
            1,
            width - 1,
            height - 1,
            self.radius,
            fill=self.fill,
            outline=self.outline,
            width=1,
            tags="panel",
        )
        self.tag_lower("panel")
        inner_width = max(1, width - 2 * self.padding)
        inner_height = max(1, height - 2 * self.padding)
        self.coords(self._window, self.padding, self.padding)
        self.itemconfigure(
            self._window, width=inner_width, height=inner_height
        )


class ActionButton(tk.Canvas):
    """Small accessible rounded button with consistent cross-platform styling."""

    def __init__(
        self,
        parent,
        text: str,
        command: Callable[[], None],
        *,
        variant: str = "secondary",
        width: int = 132,
    ) -> None:
        super().__init__(
            parent,
            width=width,
            height=38,
            bg=parent.cget("bg"),
            highlightthickness=1,
            highlightbackground=parent.cget("bg"),
            highlightcolor=Palette.ACCENT,
            bd=0,
            cursor="hand2",
            takefocus=True,
        )
        self.text = text
        self.command = command
        self.variant = variant
        self.state = tk.NORMAL
        self._hovered = False
        self.bind("<Configure>", lambda _event: self._draw())
        self.bind("<Button-1>", self._invoke)
        self.bind("<Return>", self._invoke)
        self.bind("<space>", self._invoke)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self._draw()

    def set(
        self,
        *,
        text: Optional[str] = None,
        state: Optional[str] = None,
    ) -> None:
        if text is not None:
            self.text = text
        if state is not None:
            self.state = state
            self.configure(
                cursor="hand2" if state == tk.NORMAL else "arrow"
            )
        self._draw()

    def _colors(self) -> tuple[str, str, str]:
        disabled = self.state != tk.NORMAL
        if disabled:
            return Palette.CANVAS, Palette.FAINT, Palette.BORDER
        if self.variant == "primary":
            background = (
                Palette.ACCENT_HOVER if self._hovered else Palette.ACCENT
            )
            return background, Palette.WHITE, background
        if self.variant == "danger":
            background = Palette.RED_SOFT if self._hovered else Palette.SURFACE
            return background, Palette.RED, Palette.BORDER
        background = Palette.CANVAS if self._hovered else Palette.SURFACE
        return background, Palette.TEXT, Palette.BORDER

    def _draw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), int(self.cget("width")))
        height = max(self.winfo_height(), int(self.cget("height")))
        background, foreground, outline = self._colors()
        _rounded_polygon(
            self,
            1,
            1,
            width - 1,
            height - 1,
            9,
            fill=background,
            outline=outline,
            width=1,
        )
        self.create_text(
            width / 2,
            height / 2,
            text=self.text,
            fill=foreground,
            font=("Helvetica Neue", 11, "bold"),
        )

    def _invoke(self, _event=None) -> None:
        if self.state == tk.NORMAL:
            self.command()

    def _on_enter(self, _event=None) -> None:
        self._hovered = True
        self._draw()

    def _on_leave(self, _event=None) -> None:
        self._hovered = False
        self._draw()


class ToggleSwitch(tk.Frame):
    """Compact labeled switch backed by a BooleanVar."""

    def __init__(
        self,
        parent,
        text: str,
        variable: tk.BooleanVar,
        *,
        command: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(parent, bg=parent.cget("bg"))
        self.variable = variable
        self.command = command
        self.enabled = True
        self.canvas = tk.Canvas(
            self,
            width=34,
            height=20,
            bg=self.cget("bg"),
            highlightthickness=0,
            bd=0,
            cursor="hand2",
            takefocus=True,
        )
        self.canvas.pack(side=tk.LEFT)
        self.label = tk.Label(
            self,
            text=text,
            bg=self.cget("bg"),
            fg=Palette.TEXT,
            font=("Helvetica Neue", 10),
            cursor="hand2",
        )
        self.label.pack(side=tk.LEFT, padx=(7, 0))
        for widget in (self.canvas, self.label):
            widget.bind("<Button-1>", self._toggle)
        self.canvas.bind("<Return>", self._toggle)
        self.canvas.bind("<space>", self._toggle)
        self.variable.trace_add("write", lambda *_args: self._draw())
        self._draw()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        cursor = "hand2" if enabled else "arrow"
        self.canvas.configure(cursor=cursor)
        self.label.configure(
            cursor=cursor,
            fg=Palette.TEXT if enabled else Palette.FAINT,
        )
        self._draw()

    def _toggle(self, _event=None) -> None:
        if not self.enabled:
            return
        self.variable.set(not self.variable.get())
        if self.command is not None:
            self.command()

    def _draw(self) -> None:
        self.canvas.delete("all")
        selected = self.variable.get()
        if not self.enabled:
            track = Palette.BORDER
            knob = Palette.SURFACE
        else:
            track = Palette.ACCENT if selected else "#CED1CD"
            knob = Palette.WHITE
        _rounded_polygon(
            self.canvas,
            1,
            2,
            33,
            18,
            8,
            fill=track,
            outline=track,
        )
        center_x = 24 if selected else 10
        self.canvas.create_oval(
            center_x - 6,
            4,
            center_x + 6,
            16,
            fill=knob,
            outline=knob,
        )


class VideoPanel:
    """Responsive 16:9 media card with title, status, metadata, and overlays."""

    def __init__(self, parent, title: str) -> None:
        self.panel = RoundedPanel(parent, radius=10, padding=12)
        content = self.panel.content
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(1, weight=1)

        header = tk.Frame(content, bg=Palette.SURFACE)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 9))
        header.grid_columnconfigure(0, weight=1)
        tk.Label(
            header,
            text=title,
            bg=Palette.SURFACE,
            fg=Palette.TEXT,
            font=("Helvetica Neue", 10, "bold"),
        ).grid(row=0, column=0, sticky="w")
        self.status_text = tk.StringVar(value="IDLE")
        self.status_label = tk.Label(
            header,
            textvariable=self.status_text,
            bg=Palette.SURFACE,
            fg=Palette.MUTED,
            font=("Helvetica Neue", 9, "bold"),
        )
        self.status_label.grid(row=0, column=1, sticky="e")

        self.media = tk.Frame(content, bg=Palette.VIDEO)
        self.media.grid(row=1, column=0, sticky="nsew")
        self.media.grid_rowconfigure(0, weight=1)
        self.media.grid_columnconfigure(0, weight=1)
        self.media_label = tk.Label(self.media, bg=Palette.VIDEO, bd=0)
        self.media_label.grid(row=0, column=0, sticky="nsew")
        self.overlay = tk.Label(
            self.media,
            text="",
            bg=Palette.VIDEO,
            fg=Palette.VIDEO_MUTED,
            font=("Helvetica Neue", 12),
            justify=tk.CENTER,
            padx=22,
            pady=14,
        )

        self.meta_text = tk.StringVar(value="—")
        tk.Label(
            content,
            textvariable=self.meta_text,
            bg=Palette.SURFACE,
            fg=Palette.MUTED,
            font=("Helvetica Neue", 9),
            anchor=tk.W,
        ).grid(row=2, column=0, sticky="ew", pady=(9, 0))

        self._last_frame: Optional[np.ndarray] = None
        self._render_pending = False
        self.media_label.bind("<Configure>", self._schedule_render)

    def set_status(self, text: str, color: str = Palette.MUTED) -> None:
        self.status_text.set(text)
        self.status_label.configure(fg=color)

    def set_meta(self, text: str) -> None:
        self.meta_text.set(text)

    def show_overlay(
        self,
        text: str,
        *,
        tone: str = "neutral",
    ) -> None:
        colors = {
            "neutral": (Palette.VIDEO, Palette.VIDEO_MUTED),
            "warning": ("#342F27", "#E8D6B6"),
            "error": ("#352A29", "#E5C4C0"),
        }
        background, foreground = colors[tone]
        self.overlay.configure(
            text=text,
            bg=background,
            fg=foreground,
        )
        self.overlay.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.overlay.lift()

    def hide_overlay(self) -> None:
        self.overlay.place_forget()

    def set_frame(self, frame: np.ndarray) -> None:
        self._last_frame = frame.copy()
        self._schedule_render()

    def clear_frame(self) -> None:
        self._last_frame = None
        self.media_label.configure(image="")
        self.media_label.image = None

    def _schedule_render(self, _event=None) -> None:
        if self._render_pending:
            return
        self._render_pending = True
        self.media_label.after_idle(self._render)

    def _render(self) -> None:
        self._render_pending = False
        if self._last_frame is None:
            return
        width = max(self.media_label.winfo_width(), 120)
        height = max(self.media_label.winfo_height(), 68)
        rgb = cv2.cvtColor(self._last_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), Palette.VIDEO)
        offset = ((width - image.width) // 2, (height - image.height) // 2)
        canvas.paste(image, offset)
        photo = ImageTk.PhotoImage(canvas)
        self.media_label.configure(image=photo)
        self.media_label.image = photo


class LivePoseWorker:
    """Run webcam inference off the Tk main thread and keep only fresh frames."""

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
            target=self._run, name="live-pose-worker", daemon=True
        )
        self._thread.start()

    def submit(self, frame_id: int, frame: np.ndarray) -> None:
        item = (frame_id, frame.copy())
        try:
            self._inputs.put_nowait(item)
        except queue.Full:
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
        self._thread.join(timeout=1.5)

    def _publish(
        self, payload: tuple[int, Optional[Pose], Optional[str]]
    ) -> None:
        try:
            self._outputs.put_nowait(payload)
        except queue.Full:
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
            self._publish((-1, None, f"Unable to load pose model: {exc}"))
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


class PoseApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Dance Alignment")
        self.root.geometry("1200x760")
        self.root.minsize(1040, 680)
        self.root.configure(bg=Palette.CANVAS)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.video_path: Optional[Path] = None
        self.selected_duration = 0.0
        self.reference: Optional[ReferenceSequence] = None
        self.aligner: Optional[TemporalAligner] = None
        self.window_scorer = None
        self.reference_capture: Optional[cv2.VideoCapture] = None
        self.camera_capture: Optional[cv2.VideoCapture] = None
        self.live_worker: Optional[LivePoseWorker] = None
        self.analysis_future: Optional[Future] = None
        self.analysis_cancel = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.progress_events: queue.Queue[tuple[int, int]] = queue.Queue()

        self.ui_state = UIState.EMPTY
        self.running = False
        self.after_id: Optional[str] = None
        self.feedback_animation_id: Optional[str] = None
        self.start_time = 0.0
        self.reference_frame_index = -1
        self.reference_frame: Optional[np.ndarray] = None
        self.live_frame_id = 0
        self.last_scored_frame_id = -1
        self.live_pose: Optional[Pose] = None
        self.previous_score_pose: Optional[Pose] = None
        self.received_pose_result = False
        self.pose_missing_frames = 0
        self.camera_failures = 0
        self.score_history: deque[float] = deque(maxlen=6000)
        self.pose_result_times: deque[float] = deque(maxlen=30)
        self.feedback_counts = {
            "Perfect!": 0,
            "Super!": 0,
            "Good": 0,
            "Miss": 0,
        }
        self.last_feedback = ""

        self.show_reference = tk.BooleanVar(value=True)
        self.mirror_match = tk.BooleanVar(value=True)
        self.status_text = tk.StringVar(value="Choose a reference video")
        self.camera_status_text = tk.StringVar(value="Camera · Off")
        self.score_value_text = tk.StringVar(value="—")
        self.feedback_text = tk.StringVar(value="READY")
        self.average_value_text = tk.StringVar(value="—")
        self.lag_value_text = tk.StringVar(value="—")
        self.coverage_text = tk.StringVar(value="Coverage —")
        self.timing_text = tk.StringVar(value="Timing —")
        self.metrics_text = tk.StringVar(
            value="Angle —   Shape —   Motion —   FPS —"
        )

        self._build_ui()
        self._set_state(UIState.EMPTY)
        if not MODEL_PATH.is_file():
            self.status_text.set("Pose model is missing")

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        style.configure(
            "Quiet.Horizontal.TProgressbar",
            troughcolor=Palette.CANVAS,
            background=Palette.ACCENT,
            bordercolor=Palette.CANVAS,
            lightcolor=Palette.ACCENT,
            darkcolor=Palette.ACCENT,
            thickness=3,
        )

        self.shell = tk.Frame(
            self.root,
            bg=Palette.CANVAS,
            padx=24,
            pady=16,
        )
        self.shell.pack(fill=tk.BOTH, expand=True)
        self.shell.grid_columnconfigure(0, weight=1)
        self.shell.grid_rowconfigure(2, weight=1)

        header = tk.Frame(self.shell, bg=Palette.CANVAS)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header.grid_columnconfigure(1, weight=1)
        title_block = tk.Frame(header, bg=Palette.CANVAS)
        title_block.grid(row=0, column=0, sticky="w")
        tk.Label(
            title_block,
            text="Dance Alignment",
            bg=Palette.CANVAS,
            fg=Palette.TEXT,
            font=("Helvetica Neue", 18, "bold"),
        ).pack(anchor=tk.W)
        tk.Label(
            title_block,
            text="NUS Visual Computing · Bonus Level",
            bg=Palette.CANVAS,
            fg=Palette.MUTED,
            font=("Helvetica Neue", 9),
        ).pack(anchor=tk.W, pady=(2, 0))

        status_block = tk.Frame(header, bg=Palette.CANVAS)
        status_block.grid(row=0, column=2, sticky="e")
        self.status_label = tk.Label(
            status_block,
            textvariable=self.status_text,
            bg=Palette.CANVAS,
            fg=Palette.TEXT,
            font=("Helvetica Neue", 10, "bold"),
            anchor=tk.E,
        )
        self.status_label.pack(anchor=tk.E)
        camera_line = tk.Frame(status_block, bg=Palette.CANVAS)
        camera_line.pack(anchor=tk.E, pady=(3, 0))
        self.camera_dot = tk.Canvas(
            camera_line,
            width=8,
            height=8,
            bg=Palette.CANVAS,
            highlightthickness=0,
        )
        self.camera_dot.pack(side=tk.LEFT, padx=(0, 6))
        self.camera_dot_id = self.camera_dot.create_oval(
            1, 1, 7, 7, fill=Palette.FAINT, outline=""
        )
        tk.Label(
            camera_line,
            textvariable=self.camera_status_text,
            bg=Palette.CANVAS,
            fg=Palette.MUTED,
            font=("Helvetica Neue", 9),
        ).pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(
            self.shell,
            orient=tk.HORIZONTAL,
            mode="determinate",
            style="Quiet.Horizontal.TProgressbar",
        )
        self.progress.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        self.workspace = tk.Frame(self.shell, bg=Palette.CANVAS)
        self.workspace.grid(row=2, column=0, sticky="nsew")
        self.workspace.grid_columnconfigure(0, weight=1, uniform="video")
        self.workspace.grid_columnconfigure(1, weight=1, uniform="video")
        self.workspace.grid_rowconfigure(0, weight=1)

        self.reference_panel = VideoPanel(self.workspace, "REFERENCE")
        self.reference_panel.panel.grid(
            row=0, column=0, sticky="nsew", padx=(0, 6)
        )
        self.camera_panel = VideoPanel(self.workspace, "YOU")
        self.camera_panel.panel.grid(
            row=0, column=1, sticky="nsew", padx=(6, 0)
        )

        self.score_card = RoundedPanel(
            self.shell,
            radius=10,
            padding=14,
            height=118,
        )
        self.score_card.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        score = self.score_card.content
        score.grid_columnconfigure(5, weight=1)

        self._metric_block(
            score,
            0,
            "SCORE",
            self.score_value_text,
            value_size=30,
        )
        self._separator(score, 1)
        self.feedback_label = tk.Label(
            score,
            textvariable=self.feedback_text,
            bg=Palette.SURFACE,
            fg=Palette.ACCENT,
            font=("Helvetica Neue", 20, "bold"),
            width=10,
            anchor=tk.W,
        )
        self.feedback_label.grid(
            row=0, column=2, rowspan=2, sticky="w", padx=(18, 24)
        )
        self._metric_block(
            score,
            3,
            "AVERAGE",
            self.average_value_text,
            value_size=17,
        )
        self._metric_block(
            score,
            4,
            "LAG",
            self.lag_value_text,
            value_size=17,
        )

        diagnostics = tk.Frame(score, bg=Palette.SURFACE)
        diagnostics.grid(
            row=0,
            column=5,
            rowspan=2,
            sticky="nse",
            padx=(22, 4),
        )
        top_diagnostics = tk.Frame(diagnostics, bg=Palette.SURFACE)
        top_diagnostics.pack(anchor=tk.E)
        tk.Label(
            top_diagnostics,
            textvariable=self.coverage_text,
            bg=Palette.SURFACE,
            fg=Palette.TEXT,
            font=("Helvetica Neue", 10, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 18))
        tk.Label(
            top_diagnostics,
            textvariable=self.timing_text,
            bg=Palette.SURFACE,
            fg=Palette.TEXT,
            font=("Helvetica Neue", 10, "bold"),
        ).pack(side=tk.LEFT)
        tk.Label(
            diagnostics,
            textvariable=self.metrics_text,
            bg=Palette.SURFACE,
            fg=Palette.MUTED,
            font=("Helvetica Neue", 9),
            anchor=tk.E,
        ).pack(anchor=tk.E, pady=(10, 0))

        self.control_card = RoundedPanel(
            self.shell,
            radius=10,
            padding=12,
            height=68,
        )
        self.control_card.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        controls = self.control_card.content
        controls.grid_columnconfigure(4, weight=1)

        self.open_button = ActionButton(
            controls,
            "Open Video",
            self.open_video,
            width=120,
        )
        self.open_button.grid(row=0, column=0, padx=(0, 8), sticky="w")
        self.analyze_button = ActionButton(
            controls,
            "Analyze Reference",
            self.analyze_reference,
            width=158,
        )
        self.analyze_button.grid(row=0, column=1, padx=8, sticky="w")
        self.start_button = ActionButton(
            controls,
            "Start Dance",
            self.start_dance,
            variant="primary",
            width=132,
        )
        self.start_button.grid(row=0, column=2, padx=8, sticky="w")
        self.stop_button = ActionButton(
            controls,
            "Stop",
            self.stop_dance,
            variant="danger",
            width=88,
        )
        self.stop_button.grid(row=0, column=3, padx=(8, 0), sticky="w")

        switches = tk.Frame(controls, bg=Palette.SURFACE)
        switches.grid(row=0, column=5, sticky="e")
        self.show_toggle = ToggleSwitch(
            switches,
            "Show video",
            self.show_reference,
            command=self._handle_show_reference_toggle,
        )
        self.show_toggle.pack(side=tk.LEFT, padx=(0, 22))
        self.mirror_toggle = ToggleSwitch(
            switches,
            "Mirror choreography",
            self.mirror_match,
        )
        self.mirror_toggle.pack(side=tk.LEFT)

    def _metric_block(
        self,
        parent: tk.Frame,
        column: int,
        title: str,
        variable: tk.StringVar,
        *,
        value_size: int,
    ) -> None:
        block = tk.Frame(parent, bg=Palette.SURFACE)
        block.grid(
            row=0,
            column=column,
            rowspan=2,
            sticky="w",
            padx=(4, 22),
        )
        tk.Label(
            block,
            text=title,
            bg=Palette.SURFACE,
            fg=Palette.MUTED,
            font=("Helvetica Neue", 8, "bold"),
        ).pack(anchor=tk.W)
        tk.Label(
            block,
            textvariable=variable,
            bg=Palette.SURFACE,
            fg=Palette.TEXT,
            font=("Helvetica Neue", value_size, "bold"),
        ).pack(anchor=tk.W, pady=(3, 0))

    def _separator(self, parent: tk.Frame, column: int) -> None:
        tk.Frame(
            parent,
            bg=Palette.BORDER,
            width=1,
        ).grid(
            row=0,
            column=column,
            rowspan=2,
            sticky="ns",
            padx=(0, 22),
        )

    def _set_state(
        self,
        state: UIState,
        *,
        message: Optional[str] = None,
    ) -> None:
        self.ui_state = state
        state_copy = {
            UIState.EMPTY: ("Choose a reference video", "Camera · Off"),
            UIState.VIDEO_LOADED: ("Video loaded", "Camera · Off"),
            UIState.ANALYZING: ("Analyzing reference", "Camera · Off"),
            UIState.READY: ("Reference ready", "Camera · Ready"),
            UIState.RUNNING: ("Session running", "Camera · Live"),
            UIState.POSE_LOST: ("Pose not visible", "Camera · Adjust position"),
            UIState.CAMERA_ERROR: ("Camera unavailable", "Camera · Error"),
            UIState.FINISHED: ("Session finished", "Camera · Off"),
        }
        default_message, camera_message = state_copy[state]
        self.status_text.set(message or default_message)
        self.camera_status_text.set(camera_message)
        dot_color = {
            UIState.READY: Palette.ACCENT,
            UIState.RUNNING: Palette.ACCENT,
            UIState.POSE_LOST: Palette.AMBER,
            UIState.CAMERA_ERROR: Palette.RED,
        }.get(state, Palette.FAINT)
        self.camera_dot.itemconfigure(self.camera_dot_id, fill=dot_color)

        busy = state in (UIState.ANALYZING, UIState.RUNNING, UIState.POSE_LOST)
        self.open_button.set(state=tk.DISABLED if busy else tk.NORMAL)
        can_analyze = (
            self.video_path is not None
            and state
            not in (UIState.ANALYZING, UIState.RUNNING, UIState.POSE_LOST)
        )
        self.analyze_button.set(
            state=tk.NORMAL if can_analyze else tk.DISABLED
        )
        can_start = state in (
            UIState.READY,
            UIState.CAMERA_ERROR,
            UIState.FINISHED,
        )
        start_text = {
            UIState.CAMERA_ERROR: "Retry Camera",
            UIState.FINISHED: "Dance Again",
        }.get(state, "Start Dance")
        self.start_button.set(
            text=start_text,
            state=tk.NORMAL if can_start else tk.DISABLED,
        )
        self.stop_button.set(
            state=(
                tk.NORMAL
                if state in (UIState.RUNNING, UIState.POSE_LOST)
                else tk.DISABLED
            )
        )
        self.show_toggle.set_enabled(
            self.video_path is not None and state != UIState.ANALYZING
        )
        self.mirror_toggle.set_enabled(
            state
            not in (UIState.EMPTY, UIState.VIDEO_LOADED, UIState.ANALYZING)
        )

        if state == UIState.EMPTY:
            self.reference_panel.set_status("IDLE")
            self.reference_panel.set_meta("No reference selected")
            self.reference_panel.show_overlay(
                "Choose a reference dance video\nOpen Video to begin"
            )
            self.camera_panel.set_status("OFF")
            self.camera_panel.set_meta("Camera starts with the session")
            self.camera_panel.show_overlay(
                "Camera preview\nstarts with the session"
            )
        elif state == UIState.VIDEO_LOADED:
            self.reference_panel.set_status("LOADED", Palette.BLUE)
            self.camera_panel.set_status("OFF")
            self.camera_panel.set_meta("Analyze the reference before starting")
            self.camera_panel.show_overlay("Camera ready after analysis")
        elif state == UIState.ANALYZING:
            self.reference_panel.set_status("ANALYZING", Palette.BLUE)
            self.camera_panel.set_status("OFF")
        elif state == UIState.READY:
            self.reference_panel.set_status("READY", Palette.ACCENT)
            self.reference_panel.hide_overlay()
            self.camera_panel.set_status("READY", Palette.ACCENT)
            self.camera_panel.set_meta("Camera is ready to start")
            self.camera_panel.show_overlay("Ready when you are")
        elif state == UIState.RUNNING:
            self.reference_panel.set_status("PLAYING", Palette.ACCENT)
            self.reference_panel.hide_overlay()
            self.camera_panel.set_status("TRACKING", Palette.ACCENT)
            self.camera_panel.hide_overlay()
        elif state == UIState.POSE_LOST:
            self.camera_panel.set_status("POSE LOST", Palette.AMBER)
            self.camera_panel.set_meta("Scoring paused · body not visible")
            self.camera_panel.show_overlay(
                "Step back — full body not visible\nScoring is paused",
                tone="warning",
            )
        elif state == UIState.CAMERA_ERROR:
            self.camera_panel.set_status("ERROR", Palette.RED)
            self.camera_panel.set_meta("Check camera permission or other apps")
            self.camera_panel.show_overlay(
                "Camera unavailable\nCheck permission, then Retry Camera",
                tone="error",
            )
        elif state == UIState.FINISHED:
            self.reference_panel.set_status("FINISHED", Palette.MUTED)
            self.camera_panel.set_status("FINISHED", Palette.MUTED)
            self.reference_panel.hide_overlay()
            self.camera_panel.hide_overlay()

    def _reset_score_display(self) -> None:
        self.score_value_text.set("—")
        self.feedback_text.set("READY")
        self.feedback_label.configure(fg=Palette.ACCENT)
        self.average_value_text.set("—")
        self.lag_value_text.set("—")
        self.coverage_text.set("Coverage —")
        self.timing_text.set("Timing —")
        self.metrics_text.set("Angle —   Shape —   Motion —   FPS —")
        self.last_feedback = ""

    def open_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose reference dance video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.mkv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._stop_runtime()
        self.video_path = Path(path).resolve()
        self.reference = None
        self.aligner = None
        self.progress["value"] = 0
        self._reset_score_display()
        if not self._show_video_preview():
            self.video_path = None
            self._set_state(
                UIState.EMPTY, message="Unable to read the selected video"
            )
            return
        self._set_state(
            UIState.VIDEO_LOADED,
            message=f"Loaded · {self.video_path.name}",
        )

    def _show_video_preview(self) -> bool:
        if self.video_path is None:
            return False
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            return False
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            return False
        self.selected_duration = (
            frame_count / fps if fps > 1e-3 and frame_count > 0 else 0.0
        )
        self.reference_panel.set_frame(frame)
        self.reference_panel.hide_overlay()
        self.reference_panel.set_meta(
            f"{self.video_path.name} · "
            f"{self._format_time(self.selected_duration)}"
        )
        return True

    def _handle_show_reference_toggle(self) -> None:
        if self.running:
            return
        if self.show_reference.get():
            if self.video_path is not None:
                self._show_video_preview()
        else:
            self.reference_panel.show_overlay(
                "Original video hidden\nSkeleton remains visible in the session"
            )

    def analyze_reference(self) -> None:
        if self.video_path is None:
            return
        if not MODEL_PATH.is_file():
            messagebox.showerror("Model missing", f"Cannot find:\n{MODEL_PATH}")
            return
        if self.analysis_future is not None and not self.analysis_future.done():
            return

        self._stop_runtime()
        self.analysis_cancel.clear()
        self.progress["value"] = 0
        self._set_state(UIState.ANALYZING)

        def report(current: int, total: int) -> None:
            self.progress_events.put((current, total))

        self.analysis_future = self.executor.submit(
            analyze_reference_video,
            self.video_path,
            MODEL_PATH,
            CACHE_DIR,
            progress=report,
            cancel_event=self.analysis_cancel,
        )
        self.root.after(80, self._poll_analysis)

    def _poll_analysis(self) -> None:
        future = self.analysis_future
        if future is None:
            return
        while True:
            try:
                current, total = self.progress_events.get_nowait()
            except queue.Empty:
                break
            if total > 0:
                self.progress.configure(mode="determinate")
                self.progress["maximum"] = total
                self.progress["value"] = current
                percent = current / total * 100.0
                self.status_text.set(f"Analyzing · {percent:.0f}%")
                self.reference_panel.set_meta(
                    f"Analyzing frame {current} of {total}"
                )
            else:
                self.progress.configure(mode="indeterminate")
                self.progress.start(12)
                self.reference_panel.set_meta("Reading reference frames")

        if not future.done():
            self.root.after(80, self._poll_analysis)
            return

        self.progress.stop()
        self.progress.configure(mode="determinate")
        try:
            self.reference = future.result()
        except ReferenceAnalysisCancelled:
            self._set_state(
                UIState.VIDEO_LOADED,
                message="Reference analysis cancelled",
            )
            return
        except Exception as exc:
            self._set_state(
                UIState.VIDEO_LOADED,
                message="Reference analysis failed",
            )
            messagebox.showerror("Analysis failed", str(exc))
            return

        self.aligner = TemporalAligner(
            self.reference.poses,
            self.reference.timestamps,
            search_window_seconds=0.45,
        )
        body_coverage = [
            float(np.mean(valid_keypoints(pose)[5:17]))
            for pose in self.reference.poses
        ]
        average_coverage = float(np.mean(body_coverage)) * 100.0
        source = "cache" if self.reference.from_cache else "new analysis"
        self.progress["maximum"] = self.reference.frame_count
        self.progress["value"] = self.reference.frame_count
        self.reference_panel.set_meta(
            f"{self.reference.frame_count} frames · "
            f"{self.reference.fps:.1f} FPS · "
            f"{average_coverage:.0f}% pose coverage · {source}"
        )
        self._set_state(
            UIState.READY,
            message=f"Reference ready · {average_coverage:.0f}% coverage",
        )

    def start_dance(self) -> None:
        if self.running:
            return
        if self.reference is None or self.aligner is None:
            return
        self._stop_runtime()
        self.reference_capture = cv2.VideoCapture(self.reference.source_path)
        self.camera_capture = cv2.VideoCapture(0)
        if not self.reference_capture.isOpened():
            self._release_captures()
            self._set_state(
                UIState.READY, message="Unable to reopen the reference video"
            )
            return
        if not self.camera_capture.isOpened():
            self._release_captures()
            self._set_state(UIState.CAMERA_ERROR)
            return

        self.camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.live_worker = LivePoseWorker(MODEL_PATH)
        self.running = True
        self.start_time = time.perf_counter()
        self.reference_frame_index = -1
        self.reference_frame = None
        self.live_frame_id = 0
        self.last_scored_frame_id = -1
        self.live_pose = None
        self.previous_score_pose = None
        self.received_pose_result = False
        self.pose_missing_frames = 0
        self.camera_failures = 0
        self.score_history.clear()
        self.pose_result_times.clear()
        for key in self.feedback_counts:
            self.feedback_counts[key] = 0
        self._reset_score_display()
        self.aligner.reset()
        window_frames = max(5, int(self.reference.fps * 0.5))
        self.window_scorer = WindowScorer(window_size=window_frames, punish_threshold=0.35)
        self._set_state(UIState.RUNNING)
        self.status_text.set("Starting pose tracking")
        self.feedback_text.set("GO")
        self.camera_panel.show_overlay("Starting pose tracking…")
        self._tick()

    def stop_dance(self) -> None:
        had_scores = bool(self.score_history)
        self._stop_runtime()
        if had_scores:
            self._show_finished_state(completed=False)
        elif self.reference is not None:
            self._set_state(UIState.READY)
        else:
            self._set_state(
                UIState.VIDEO_LOADED
                if self.video_path is not None
                else UIState.EMPTY
            )

    def _stop_runtime(self) -> None:
        self.running = False
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None
        if self.feedback_animation_id is not None:
            try:
                self.root.after_cancel(self.feedback_animation_id)
            except tk.TclError:
                pass
            self.feedback_animation_id = None
        if self.live_worker is not None:
            self.live_worker.stop()
            self.live_worker = None
        self._release_captures()
        if self.window_scorer is not None:
            self.window_scorer.reset()

    def _show_finished_state(self, *, completed: bool) -> None:
        average = (
            float(np.mean(self.score_history)) * 100.0
            if self.score_history
            else 0.0
        )
        self.average_value_text.set(f"{average:.1f}%")
        self.feedback_text.set("FINISHED")
        self.feedback_label.configure(fg=Palette.TEXT)
        self.coverage_text.set(
            f"Perfect {self.feedback_counts['Perfect!']}   "
            f"Super {self.feedback_counts['Super!']}"
        )
        self.timing_text.set(
            f"Good {self.feedback_counts['Good']}   "
            f"Miss {self.feedback_counts['Miss']}"
        )
        self.metrics_text.set(
            f"{len(self.score_history)} evaluated poses · "
            f"session average {average:.1f}%"
        )
        self.camera_panel.set_meta(
            f"Session average {average:.1f}% · "
            f"{len(self.score_history)} evaluated poses"
        )
        message = "Dance complete" if completed else "Session stopped"
        self._set_state(
            UIState.FINISHED,
            message=f"{message} · average {average:.1f}%",
        )

    def _release_captures(self) -> None:
        for capture in (self.reference_capture, self.camera_capture):
            if capture is not None:
                capture.release()
        self.reference_capture = None
        self.camera_capture = None

    def _read_reference_frame(self, target_index: int) -> Optional[np.ndarray]:
        capture = self.reference_capture
        if capture is None:
            return None
        if target_index < self.reference_frame_index:
            capture.set(cv2.CAP_PROP_POS_FRAMES, target_index)
            self.reference_frame_index = target_index - 1
        while self.reference_frame_index < target_index:
            ok, frame = capture.read()
            if not ok:
                return None
            self.reference_frame = frame
            self.reference_frame_index += 1
        return self.reference_frame.copy() if self.reference_frame is not None else None

    def _tick(self) -> None:
        if not self.running or self.reference is None or self.aligner is None:
            return
        now = time.perf_counter()
        elapsed = now - self.start_time
        expected_index = min(
            int(elapsed * self.reference.fps), self.reference.frame_count - 1
        )
        if elapsed > self.reference.duration + 1.0 / self.reference.fps:
            self._stop_runtime()
            self._show_finished_state(completed=True)
            return

        reference_frame = self._read_reference_frame(expected_index)
        if reference_frame is None:
            self._stop_runtime()
            self._show_finished_state(completed=True)
            return
        reference_pose = self.reference.poses[expected_index]
        if self.show_reference.get():
            reference_display = reference_frame
        else:
            reference_display = np.full_like(reference_frame, 245)
        reference_display = draw_pose(
            reference_display,
            reference_pose,
            line_color=(112, 124, 97),
            point_color=(236, 241, 238),
        )
        self.reference_panel.set_frame(reference_display)
        self.reference_panel.set_meta(
            f"{self._format_time(elapsed)} / "
            f"{self._format_time(self.reference.duration)} · "
            f"frame {expected_index + 1}"
        )

        camera_ok, camera_frame = (
            self.camera_capture.read()
            if self.camera_capture is not None
            else (False, None)
        )
        if not camera_ok or camera_frame is None:
            self.camera_failures += 1
            if self.camera_failures >= 5:
                self._stop_runtime()
                self._set_state(UIState.CAMERA_ERROR)
                return
            self.after_id = self.root.after(60, self._tick)
            return
        self.camera_failures = 0
        camera_frame = cv2.flip(camera_frame, 1)
        self.live_frame_id += 1

        output = None
        if self.live_worker is not None:
            self.live_worker.submit(self.live_frame_id, camera_frame)
            output = self.live_worker.latest()

        if output is not None:
            result_frame_id, pose, error = output
            if error:
                self._stop_runtime()
                self._set_state(UIState.CAMERA_ERROR, message=error)
                return
            self.received_pose_result = True
            self.pose_result_times.append(now)
            self.live_pose = pose
            scoreable = self._is_scoreable_pose(pose)
            if scoreable:
                self.pose_missing_frames = 0
                if self.ui_state == UIState.POSE_LOST:
                    self._set_state(UIState.RUNNING)
                else:
                    self.camera_panel.hide_overlay()
                if result_frame_id != self.last_scored_frame_id:
                    score_pose = mirror_pose(pose) if self.mirror_match.get() else pose
                    ref_pose = self.reference.poses[expected_index]
                    if self.window_scorer is not None:
                        self.window_scorer.add_frame(ref_pose, score_pose)
                        window_breakdown = self.window_scorer.compute_window_score()
                        if window_breakdown is not None:
                            self.previous_score_pose = score_pose
                            self.last_scored_frame_id = result_frame_id
                            self._update_score(window_breakdown)
                    else:
                        # fallback to single-frame scoring
                        alignment = self.aligner.align(score_pose, elapsed, previous_user_pose=self.previous_score_pose)
                        self.previous_score_pose = score_pose
                        self.last_scored_frame_id = result_frame_id
                        self._update_score(alignment.breakdown)
            else:
                self.pose_missing_frames += 1
                if self.pose_missing_frames >= 3:
                    self._set_state(UIState.POSE_LOST)
        elif not self.received_pose_result:
            self.camera_panel.set_meta("Loading pose model…")

        camera_display = draw_pose(
            camera_frame,
            self.live_pose,
            line_color=(138, 117, 97),
            point_color=(235, 231, 225),
        )
        self.camera_panel.set_frame(camera_display)
        if self.received_pose_result and self.ui_state == UIState.RUNNING:
            visible = (
                int(np.count_nonzero(valid_keypoints(self.live_pose)[5:17]))
                if self.live_pose is not None
                else 0
            )
            self.camera_panel.set_meta(
                f"Pose detected · {visible}/12 body joints · "
                f"{self._pose_fps():.1f} FPS"
            )
        self.after_id = self.root.after(10, self._tick)

    @staticmethod
    def _is_scoreable_pose(pose: Optional[Pose]) -> bool:
        if pose is None:
            return False
        body = valid_keypoints(pose)[5:17]
        torso = valid_keypoints(pose)[5:13]
        return bool(np.count_nonzero(body) >= 4 and np.count_nonzero(torso) >= 2)

    def _pose_fps(self) -> float:
        if len(self.pose_result_times) < 2:
            return 0.0
        duration = self.pose_result_times[-1] - self.pose_result_times[0]
        if duration <= 0:
            return 0.0
        return (len(self.pose_result_times) - 1) / duration

    def _update_score(self, breakdown: ScoreBreakdown) -> None:
        self.score_history.append(breakdown.total)
        score = breakdown.total * 100.0
        average = float(np.mean(self.score_history)) * 100.0
        feedback, _bgr = map_score_to_feedback(breakdown.total)
        self.feedback_counts[feedback] += 1

        self.score_value_text.set(f"{score:.0f}")
        self.average_value_text.set(f"{average:.1f}%")
        lag_prefix = "+" if breakdown.lag_ms > 0 else "−"
        if abs(breakdown.lag_ms) < 0.5:
            lag_prefix = ""
        self.lag_value_text.set(
            f"{lag_prefix}{abs(breakdown.lag_ms):.0f} ms"
        )
        self.coverage_text.set(
            f"Coverage {breakdown.coverage * 100:.0f}%"
        )
        if abs(breakdown.lag_ms) < 90:
            timing = "Timing aligned"
        elif breakdown.lag_ms > 0:
            timing = "Timing ahead"
        else:
            timing = "Timing behind"
        self.timing_text.set(timing)
        motion_text = (
            f"{breakdown.motion * 100:.0f}%"
            if breakdown.motion is not None
            else "—"
        )
        self.metrics_text.set(
            f"Angle {breakdown.angle * 100:.0f}%   "
            f"Shape {breakdown.position * 100:.0f}%   "
            f"Motion {motion_text}   FPS {self._pose_fps():.1f}"
        )
        self.feedback_text.set(feedback.upper().rstrip("!"))
        color = FEEDBACK_COLORS[feedback]
        if feedback != self.last_feedback:
            self.last_feedback = feedback
            self._animate_feedback(color)
        else:
            self.feedback_label.configure(fg=color)

    def _animate_feedback(self, target: str) -> None:
        if self.feedback_animation_id is not None:
            try:
                self.root.after_cancel(self.feedback_animation_id)
            except tk.TclError:
                pass
        start_rgb = self._hex_to_rgb(Palette.BORDER)
        target_rgb = self._hex_to_rgb(target)
        steps = 5

        def step(index: int) -> None:
            ratio = index / steps
            rgb = tuple(
                round(start + (end - start) * ratio)
                for start, end in zip(start_rgb, target_rgb)
            )
            self.feedback_label.configure(
                fg=f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            )
            if index < steps:
                self.feedback_animation_id = self.root.after(
                    36, step, index + 1
                )
            else:
                self.feedback_animation_id = None

        step(0)

    @staticmethod
    def _hex_to_rgb(value: str) -> tuple[int, int, int]:
        value = value.lstrip("#")
        return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))

    @staticmethod
    def _format_time(seconds: float) -> str:
        seconds = max(0, int(seconds))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def close(self) -> None:
        self.analysis_cancel.set()
        self._stop_runtime()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    PoseApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
