"""Centralized configuration for the "Hole in the Wall" bonus activity.

All tunables are here so they can be adjusted right before the demo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WALL_POSES_PATH = PROJECT_ROOT / "cache" / "wall_poses.json"
DEFAULT_WALL_SCHEDULE_PATH = PROJECT_ROOT / "cache" / "wall_schedule.json"

# -----------------------------------------------------------------------------
# Wall geometry & timing
# -----------------------------------------------------------------------------
WALL_APPROACH_SECONDS: float = 4.0          # time from spawn to judge line (slower for demo)
WALL_SPAWN_SCALE: float = 0.2               # initial visual scale
WALL_JUDGE_SCALE: float = 1.0               # scale at the judge line
WALL_JUDGE_Y_RATIO: float = 0.75            # judge line position (ratio of frame height)
WALL_SPAWN_Y_RATIO: float = 0.15            # spawn position (ratio of frame height)

# Visual styling (BGR for OpenCV)
WALL_COLOR: tuple[int, int, int] = (80, 80, 80)          # grey wall
WALL_GUIDE_COLOR: tuple[int, int, int] = (200, 200, 200) # silhouette edge guide
WALL_OVERLAY_ALPHA: float = 0.65
WALL_PASS_PARTICLES: int = 45
WALL_FAIL_HOLD_SECONDS: float = 0.5

# -----------------------------------------------------------------------------
# Pose extraction (extract_wall_poses.py)
# -----------------------------------------------------------------------------
MIN_BODY_KEYPOINTS: int = 8                 # min visible body joints in a target pose
MIN_POSE_CONFIDENCE: float = 0.5            # average confidence threshold
MIN_POSE_COUNT: int = 8                     # absolute minimum number of wall poses
MAX_POSE_COUNT: int = 15                    # absolute maximum number of wall poses
SIMILARITY_DEDUP_THRESHOLD: float = 0.90    # drop adjacent poses if too similar

# A frame is "still" when its local motion energy is below this percentile
STILLNESS_PERCENTILE: float = 15.0

# -----------------------------------------------------------------------------
# Wall scoring
# -----------------------------------------------------------------------------
WALL_SIMILARITY_PERFECT: float = 0.85
WALL_SIMILARITY_GOOD: float = 0.70
WALL_SCORE_PERFECT: int = 100
WALL_SCORE_GOOD: int = 50
WALL_SCORE_MISS: int = 0
WALL_SCORE_COMBO_CAP: int = 4               # max combo multiplier

# -----------------------------------------------------------------------------
# Wall spawning schedule
# -----------------------------------------------------------------------------
WALL_INTERVAL_SECONDS: float = 12.0         # fixed-interval fallback (more breathing room)
WALL_TIME_WINDOWS: list[tuple[float, float]] = [
    (8.0, 25.0),
    (38.0, 55.0),
]
# If librosa beat tracking is available and its confidence exceeds this,
# beats are used; otherwise we fall back to fixed intervals.
BEAT_TRACK_CONFIDENCE_MIN: float = 0.35

# -----------------------------------------------------------------------------
# FREEZE ("wooden man") bonus
# -----------------------------------------------------------------------------
FREEZE_EVENTS: list[dict[str, float]] = [
    {"start": 16.0, "duration": 2.5},
    {"start": 45.0, "duration": 3.0},
]
FREEZE_DISPLACEMENT_THRESHOLD: float = 120.0   # pixels summed over all joints
FREEZE_SCORE_STILL: int = 50
FREEZE_SCORE_MOVE: int = -100
FREEZE_OVERLAY_COLOR: tuple[int, int, int] = (60, 60, 220)  # red tint (BGR)

# -----------------------------------------------------------------------------
# Demo / synthetic pose generation
# -----------------------------------------------------------------------------
DEMO_OUTPUT_DIR: Path = PROJECT_ROOT / "outputs" / "wall_demo"
DEMO_SAVE_FRAMES: list[str] = ["spawn", "approach", "judge", "pass", "fail"]


def load_demo_schedule() -> Optional[list[dict]]:
    """Return a deterministic FREEZE schedule for the demo mode."""
    return [
        {"start": 3.0, "duration": 2.0},
        {"start": 9.0, "duration": 2.0},
    ]
