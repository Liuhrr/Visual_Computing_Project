"""Hole in the Wall bonus activity: state machine, rendering, and scoring.

All new game logic lives here so that danceapp.py only needs to import and
call update(). Existing pose utilities and scoring functions are reused.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from utils.pose_utils import (
    Pose,
    SKELETON,
    draw_pose,
    normalize_pose,
    valid_keypoints,
)
from utils.scoring import compute_pose_score

import wall_config as cfg


class WallState(Enum):
    IDLE = "idle"
    SPAWN = "spawn"
    APPROACH = "approach"
    JUDGE = "judge"
    PASS = "pass"
    FAIL = "fail"


@dataclass
class BonusEvent:
    kind: str          # "perfect", "good", "miss", "freeze_still", "freeze_move"
    score: int
    combo: int
    message: str


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    color: tuple[int, int, int]
    size: float


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _pose_from_dict(data: dict) -> Pose:
    xy = np.asarray(data["xy"], dtype=np.float32)
    confidence = np.asarray(data["confidence"], dtype=np.float32)
    return Pose(xy=xy, confidence=confidence)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _ease_out_quad(t: float) -> float:
    return 1.0 - (1.0 - t) * (1.0 - t)


def _put_text_centered(
    frame: np.ndarray,
    text: str,
    center: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    origin = (center[0] - w // 2, center[1] + h // 2)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Single wall
# ---------------------------------------------------------------------------
class Wall:
    """One flying wall with a human-shaped cutout."""

    def __init__(self, target_pose: Pose, spawn_time: float) -> None:
        self.target_pose = target_pose
        self.spawn_time = spawn_time
        self.judge_time = spawn_time + cfg.WALL_APPROACH_SECONDS
        self.state = WallState.SPAWN
        self.judged = False
        self.result: Optional[str] = None  # "perfect", "good", "miss"
        self.similarity = 0.0
        self.pass_time: Optional[float] = None
        self.particles: list[Particle] = []
        self._rgba: Optional[np.ndarray] = None

    def update(self, elapsed: float) -> WallState:
        if self.state == WallState.IDLE:
            return self.state

        if self.state == WallState.SPAWN and elapsed >= self.spawn_time:
            self.state = WallState.APPROACH

        if self.state == WallState.APPROACH and elapsed >= self.judge_time:
            self.state = WallState.JUDGE

        if self.state in (WallState.PASS, WallState.FAIL):
            if self.pass_time is not None:
                hold = cfg.WALL_FAIL_HOLD_SECONDS if self.state == WallState.FAIL else 0.45
                if elapsed - self.pass_time >= hold:
                    self.state = WallState.IDLE

        return self.state

    def judge(self, user_pose: Optional[Pose]) -> BonusEvent:
        """Evaluate the player's pose at the judge line."""
        self.judged = True
        self.state = WallState.JUDGE

        if user_pose is None:
            self.similarity = 0.0
            self.result = "miss"
        else:
            breakdown = compute_pose_score(self.target_pose, user_pose)
            self.similarity = breakdown.total
            if self.similarity >= cfg.WALL_SIMILARITY_PERFECT:
                self.result = "perfect"
            elif self.similarity >= cfg.WALL_SIMILARITY_GOOD:
                self.result = "good"
            else:
                self.result = "miss"

        self.pass_time = None  # set by state transition
        return self._build_event()

    def _build_event(self) -> BonusEvent:
        if self.result == "perfect":
            return BonusEvent("perfect", cfg.WALL_SCORE_PERFECT, -1, "PERFECT")
        if self.result == "good":
            return BonusEvent("good", cfg.WALL_SCORE_GOOD, -1, "GOOD")
        return BonusEvent("miss", cfg.WALL_SCORE_MISS, 0, "MISS")

    def mark_pass(self, elapsed: float) -> None:
        self.state = WallState.PASS
        self.pass_time = elapsed
        self._spawn_particles(success=True)

    def mark_fail(self, elapsed: float) -> None:
        self.state = WallState.FAIL
        self.pass_time = elapsed
        self._spawn_particles(success=False)

    def _spawn_particles(self, success: bool) -> None:
        cx, cy = 640, 360  # center of 1280x720; updated during render
        for _ in range(cfg.WALL_PASS_PARTICLES):
            angle = random.uniform(0.0, 2.0 * math.pi)
            speed = random.uniform(3.0, 12.0)
            color = (
                (80, 240, 80) if success else (80, 80, 255)
            )
            self.particles.append(
                Particle(
                    x=float(cx),
                    y=float(cy),
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed,
                    life=1.0,
                    color=color,
                    size=random.uniform(2.0, 6.0),
                )
            )

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------
    def render(
        self,
        frame: np.ndarray,
        elapsed: float,
        user_pose: Optional[Pose],
    ) -> np.ndarray:
        """Draw the wall and related feedback onto the frame."""
        if self.state == WallState.IDLE:
            return frame

        h, w = frame.shape[:2]
        wall_h, wall_w = 720, 1280

        # Compute wall transform
        if self.state == WallState.SPAWN:
            t = 0.0
        elif self.state == WallState.APPROACH:
            t = _ease_out_quad(
                min(1.0, max(0.0, (elapsed - self.spawn_time) / cfg.WALL_APPROACH_SECONDS))
            )
        else:
            t = 1.0

        scale = _lerp(cfg.WALL_SPAWN_SCALE, cfg.WALL_JUDGE_SCALE, t)
        if self.state == WallState.PASS:
            # shrink and fade after passing
            post_t = min(1.0, (elapsed - self.pass_time) / 0.45) if self.pass_time else 1.0
            scale *= (1.0 - post_t * 0.3)
        if self.state == WallState.FAIL:
            scale = cfg.WALL_JUDGE_SCALE

        spawn_y = int(cfg.WALL_SPAWN_Y_RATIO * h)
        judge_y = int(cfg.WALL_JUDGE_Y_RATIO * h)
        center_y = int(_lerp(spawn_y, judge_y, t))
        center_x = w // 2

        # Build wall RGBA once
        if self._rgba is None:
            self._rgba = self._build_wall_rgba((wall_h, wall_w))

        wall_layer = self._rgba.copy()

        # Overlay user skeleton inside the cutout while approaching
        if self.state in (WallState.APPROACH, WallState.SPAWN):
            overlay = wall_layer[:, :, :3].copy()
            if user_pose is not None:
                overlay = self._draw_user_alignment(overlay, user_pose, (wall_h, wall_w))
            alpha = wall_layer[:, :, 3:4].astype(np.float32) / 255.0
            wall_layer[:, :, :3] = overlay

        # Scale and composite onto frame
        scaled_w = max(1, int(wall_w * scale))
        scaled_h = max(1, int(wall_h * scale))
        scaled = cv2.resize(wall_layer, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        x1 = center_x - scaled_w // 2
        y1 = center_y - scaled_h // 2
        x2 = x1 + scaled_w
        y2 = y1 + scaled_h

        fx1, fy1 = max(0, x1), max(0, y1)
        fx2, fy2 = min(w, x2), min(h, y2)
        sx1 = fx1 - x1
        sy1 = fy1 - y1
        sx2 = sx1 + (fx2 - fx1)
        sy2 = sy1 + (fy2 - fy1)

        if fx2 > fx1 and fy2 > fy1:
            src = scaled[sy1:sy2, sx1:sx2]
            alpha = src[:, :, 3:4].astype(np.float32) / 255.0 * cfg.WALL_OVERLAY_ALPHA
            frame_roi = frame[fy1:fy2, fx1:fx2].astype(np.float32)
            blended = frame_roi * (1.0 - alpha) + src[:, :, :3].astype(np.float32) * alpha
            frame[fy1:fy2, fx1:fx2] = blended.astype(np.uint8)

        # Judge line
        if self.state in (WallState.APPROACH, WallState.JUDGE):
            cv2.line(frame, (0, judge_y), (w, judge_y), (0, 255, 255), 2, cv2.LINE_AA)

        # Result text
        if self.state in (WallState.PASS, WallState.FAIL, WallState.JUDGE):
            color = (80, 240, 80) if self.result in ("perfect", "good") else (80, 80, 255)
            label = self.result.upper() if self.result else ""
            _put_text_centered(frame, label, (w // 2, h // 3), 2.2, color, 4)
            score_text = f"+{self._event_score()}"
            if self.result == "miss":
                score_text = "MISS"
            _put_text_centered(frame, score_text, (w // 2, h // 3 + 70), 1.4, color, 3)

        # Particles
        self._update_and_draw_particles(frame, center_x, center_y)

        # Progress bar (overlap % while approaching)
        if self.state in (WallState.APPROACH, WallState.SPAWN):
            sim_text = f"MATCH {int(self.similarity * 100)}%"
            if user_pose is not None:
                live = compute_pose_score(self.target_pose, user_pose)
                self.similarity = live.total
                sim_text = f"MATCH {int(live.total * 100)}%"
            bar_w = int(w * 0.35)
            bar_x = (w - bar_w) // 2
            bar_y = h - 40
            fill_w = int(bar_w * self.similarity)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 14), (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + 14), (80, 240, 80), -1)
            _put_text_centered(frame, sim_text, (w // 2, bar_y - 12), 0.6, (255, 255, 255), 1)

        return frame

    def _event_score(self) -> int:
        if self.result == "perfect":
            return cfg.WALL_SCORE_PERFECT
        if self.result == "good":
            return cfg.WALL_SCORE_GOOD
        return cfg.WALL_SCORE_MISS

    def _build_wall_rgba(self, size: tuple[int, int]) -> np.ndarray:
        """Create an RGBA wall image with a human-shaped transparent cutout."""
        h, w = size
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = cfg.WALL_COLOR
        rgba[:, :, 3] = 255

        mask = np.zeros((h, w), dtype=np.uint8)

        # Project target pose into the wall coordinate system
        norm_pose = normalize_pose(self.target_pose)
        valid = valid_keypoints(norm_pose)
        if np.count_nonzero(valid) < 3:
            return rgba

        # Map normalized pose to a centered 500x700 region
        scale = 280.0
        offset = np.array([w / 2.0, h / 2.0 + 40.0], dtype=np.float32)
        points = {}
        for idx in np.flatnonzero(valid):
            points[idx] = tuple(
                np.rint(norm_pose.xy[idx] * scale + offset).astype(int)
            )

        # Draw thick skeleton as body region
        for start, end in SKELETON:
            if start in points and end in points:
                cv2.line(mask, points[start], points[end], 255, 28, cv2.LINE_AA)

        # Draw joints
        for idx, pt in points.items():
            radius = 22 if idx in (5, 6, 11, 12) else 16
            cv2.circle(mask, pt, radius, 255, -1, cv2.LINE_AA)

        # Make the body region transparent
        rgba[mask > 0, 3] = 0

        # Draw guide skeleton on the silhouette edge
        guide = np.zeros((h, w, 3), dtype=np.uint8)
        guide_pose = Pose(
            xy=np.array(
                [np.array(points.get(i, [np.nan, np.nan]), dtype=np.float32) for i in range(17)],
                dtype=np.float32,
            ),
            confidence=np.array(
                [1.0 if i in points else 0.0 for i in range(17)], dtype=np.float32
            ),
        )
        guide = draw_pose(
            guide,
            guide_pose,
            line_color=cfg.WALL_GUIDE_COLOR,
            point_color=cfg.WALL_GUIDE_COLOR,
            conf_threshold=0.5,
        )

        edge_mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
        rgba[edge_mask > 0, :3] = guide[edge_mask > 0]

        return rgba

    def _draw_user_alignment(
        self,
        wall_bgr: np.ndarray,
        user_pose: Pose,
        size: tuple[int, int],
    ) -> np.ndarray:
        """Project and draw the user's live skeleton to align with the cutout."""
        h, w = size
        norm_user = normalize_pose(user_pose)
        valid = valid_keypoints(norm_user)
        if np.count_nonzero(valid) < 3:
            return wall_bgr

        scale = 280.0
        offset = np.array([w / 2.0, h / 2.0 + 40.0], dtype=np.float32)
        aligned_xy = norm_user.xy * scale + offset
        aligned_confidence = norm_user.confidence.copy()
        aligned_pose = Pose(aligned_xy, aligned_confidence)

        overlay = wall_bgr.copy()
        overlay = draw_pose(
            overlay,
            aligned_pose,
            line_color=(80, 240, 80),
            point_color=(0, 255, 0),
            conf_threshold=0.25,
        )
        return cv2.addWeighted(wall_bgr, 0.35, overlay, 0.65, 0)

    def _update_and_draw_particles(
        self,
        frame: np.ndarray,
        center_x: int,
        center_y: int,
    ) -> None:
        if self.state not in (WallState.PASS, WallState.FAIL):
            return
        alive = []
        for p in self.particles:
            p.x += p.vx
            p.y += p.vy
            p.vy += 0.25  # gravity
            p.life -= 0.03
            if p.life > 0:
                alive.append(p)
                x = int(center_x + (p.x - 640))
                y = int(center_y + (p.y - 360))
                if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    cv2.circle(frame, (x, y), int(p.size), p.color, -1, cv2.LINE_AA)
        self.particles = alive


# ---------------------------------------------------------------------------
# Schedule generation
# ---------------------------------------------------------------------------
def _load_audio_beats(video_path: Path) -> Optional[np.ndarray]:
    """Try to extract beat timestamps with librosa; return None on failure."""
    try:
        import librosa
    except Exception:
        return None

    try:
        y, sr = librosa.load(str(video_path), sr=None, mono=True)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        if beats is None or len(beats) < 4:
            return None
        # librosa 0.10+ returns frames; older versions return sample frames
        if hasattr(librosa.frames_to_time, '__call__'):
            times = librosa.frames_to_time(beats, sr=sr)
        else:
            times = beats / sr
        if len(times) < 4:
            return None
        return np.asarray(times, dtype=np.float32)
    except Exception:
        return None


def build_wall_schedule(
    video_path: Path,
    wall_poses: list[dict],
    output_path: Optional[Path] = None,
) -> dict:
    """Build a schedule of wall spawn times.

    Prefers librosa beat tracking inside configured time windows; falls back
    to a fixed interval otherwise.
    """
    beats = _load_audio_beats(video_path)
    windows = cfg.WALL_TIME_WINDOWS

    spawn_times: list[float] = []
    source = "fixed"

    if beats is not None and len(beats) >= 4:
        # Keep beats that fall inside windows and are spaced enough apart
        used_beats = []
        last = -1.0
        for b in beats:
            if b < 0:
                continue
            in_window = any(start <= b <= end for start, end in windows)
            if in_window and (not used_beats or b - used_beats[-1] >= 1.5):
                used_beats.append(float(b))
        if len(used_beats) >= len(wall_poses):
            spawn_times = used_beats[: len(wall_poses)]
            source = "librosa"

    if not spawn_times:
        # Fixed interval fallback, anchored inside the first window
        start_time = windows[0][0] if windows else 5.0
        spawn_times = [
            start_time + i * cfg.WALL_INTERVAL_SECONDS
            for i in range(len(wall_poses))
        ]

    schedule = {
        "source": source,
        "pose_count": len(wall_poses),
        "spawn_times": spawn_times,
        "windows": windows,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(schedule, f, indent=2)

    return schedule


# ---------------------------------------------------------------------------
# Wall game manager
# ---------------------------------------------------------------------------
class WallGame:
    """Manages the wall schedule and active wall."""

    def __init__(
        self,
        wall_poses: list[dict],
        schedule: dict,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.poses = [_pose_from_dict(p) for p in wall_poses]
        self.spawn_times: list[float] = schedule.get("spawn_times", [])
        self.schedule_source = schedule.get("source", "fixed")
        self.active_wall: Optional[Wall] = None
        self.next_spawn_index = 0
        self.combo = 1
        self.pending_events: list[BonusEvent] = []

    @classmethod
    def from_files(
        cls,
        poses_path: Path = cfg.DEFAULT_WALL_POSES_PATH,
        schedule_path: Path = cfg.DEFAULT_WALL_SCHEDULE_PATH,
        enabled: bool = True,
    ) -> "WallGame":
        with open(poses_path, "r", encoding="utf-8") as f:
            poses_data = json.load(f)
        with open(schedule_path, "r", encoding="utf-8") as f:
            schedule = json.load(f)
        return cls(poses_data.get("poses", []), schedule, enabled=enabled)

    def update(
        self,
        elapsed: float,
        user_pose: Optional[Pose],
        frame: np.ndarray,
    ) -> tuple[np.ndarray, list[BonusEvent]]:
        """Advance state machine, render, and return events."""
        self.pending_events.clear()
        if not self.enabled or not self.poses:
            return frame, self.pending_events

        # Spawn next wall
        if self.active_wall is None or self.active_wall.state == WallState.IDLE:
            if self.next_spawn_index < len(self.spawn_times):
                if elapsed >= self.spawn_times[self.next_spawn_index]:
                    target = self.poses[self.next_spawn_index % len(self.poses)]
                    self.active_wall = Wall(
                        target, self.spawn_times[self.next_spawn_index]
                    )
                    self.next_spawn_index += 1

        if self.active_wall is not None:
            state = self.active_wall.update(elapsed)

            if state == WallState.JUDGE and not self.active_wall.judged:
                event = self.active_wall.judge(user_pose)
                # apply combo
                if event.kind in ("perfect", "good"):
                    event = BonusEvent(
                        event.kind,
                        event.score * self.combo,
                        min(self.combo + 1, cfg.WALL_SCORE_COMBO_CAP),
                        event.message,
                    )
                    self.combo = event.combo
                else:
                    event = BonusEvent(
                        event.kind, event.score, 1, event.message
                    )
                    self.combo = 1
                self.pending_events.append(event)

                if event.kind in ("perfect", "good"):
                    self.active_wall.mark_pass(elapsed)
                else:
                    self.active_wall.mark_fail(elapsed)

            frame = self.active_wall.render(frame, elapsed, user_pose)

        return frame, self.pending_events

    def reset(self) -> None:
        self.active_wall = None
        self.next_spawn_index = 0
        self.combo = 1
        self.pending_events.clear()


# ---------------------------------------------------------------------------
# FREEZE (wooden man) mini-game
# ---------------------------------------------------------------------------
class FreezeGame:
    """Detects whether the player stays still during configured silence events."""

    def __init__(
        self,
        events: Optional[list[dict]] = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.events = events or cfg.FREEZE_EVENTS
        self.active_index: Optional[int] = None
        self.cumulative_displacement = 0.0
        self.previous_pose: Optional[Pose] = None
        self.last_event: Optional[BonusEvent] = None
        self._finished_indices: set[int] = set()

    def update(
        self,
        elapsed: float,
        user_pose: Optional[Pose],
        frame: np.ndarray,
    ) -> tuple[np.ndarray, Optional[BonusEvent]]:
        if not self.enabled:
            return frame, None

        event_index = None
        for i, ev in enumerate(self.events):
            start = ev["start"]
            end = start + ev["duration"]
            if start <= elapsed < end:
                event_index = i
                break

        overlay = frame.copy()

        if event_index is not None and event_index not in self._finished_indices:
            if self.active_index != event_index:
                # New FREEZE event started
                self.active_index = event_index
                self.cumulative_displacement = 0.0
                self.previous_pose = None
                self.last_event = None

            # accumulate displacement
            if user_pose is not None and self.previous_pose is not None:
                curr_valid = valid_keypoints(user_pose)
                prev_valid = valid_keypoints(self.previous_pose)
                common = curr_valid & prev_valid
                if np.count_nonzero(common) > 0:
                    disp = np.linalg.norm(
                        user_pose.xy[common] - self.previous_pose.xy[common], axis=1
                    )
                    self.cumulative_displacement += float(np.sum(disp))

            self.previous_pose = user_pose

            # Visual feedback
            alpha = 0.25
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), cfg.FREEZE_OVERLAY_COLOR, -1)
            frame = cv2.addWeighted(frame, 1.0 - alpha, overlay, alpha, 0)
            _put_text_centered(frame, "FREEZE!", (frame.shape[1] // 2, 80), 2.5, (255, 255, 255), 4)
            remain = self.events[event_index]["duration"] - (
                elapsed - self.events[event_index]["start"]
            )
            _put_text_centered(
                frame,
                f"HOLD STILL {remain:.1f}s",
                (frame.shape[1] // 2, 140),
                0.9,
                (220, 220, 220),
                2,
            )
        else:
            # Event just ended
            if self.active_index is not None and self.active_index not in self._finished_indices:
                if self.cumulative_displacement < cfg.FREEZE_DISPLACEMENT_THRESHOLD:
                    event = BonusEvent(
                        "freeze_still",
                        cfg.FREEZE_SCORE_STILL,
                        -1,
                        "STILL!",
                    )
                else:
                    event = BonusEvent(
                        "freeze_move",
                        cfg.FREEZE_SCORE_MOVE,
                        -1,
                        "MOVED!",
                    )
                self.last_event = event
                self._finished_indices.add(self.active_index)
                self.active_index = None
                self.previous_pose = None
                color = (80, 240, 80) if event.kind == "freeze_still" else (60, 60, 255)
                _put_text_centered(frame, event.message, (frame.shape[1] // 2, 140), 2.4, color, 5)
                _put_text_centered(frame, f"{event.score:+,}", (frame.shape[1] // 2, 200), 1.4, color, 3)
                return frame, event

            self.active_index = None
            self.previous_pose = None

        # Show lingering result for 1.5s
        if self.last_event is not None:
            color = (80, 240, 80) if self.last_event.kind == "freeze_still" else (60, 60, 255)
            _put_text_centered(frame, self.last_event.message, (frame.shape[1] // 2, 140), 2.4, color, 5)
            score_text = f"{self.last_event.score:+,}"
            _put_text_centered(frame, score_text, (frame.shape[1] // 2, 200), 1.4, color, 3)

        return frame, None

    def reset(self) -> None:
        self.active_index = None
        self.cumulative_displacement = 0.0
        self.previous_pose = None
        self.last_event = None
        self._finished_indices.clear()
