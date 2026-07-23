from __future__ import annotations

import numpy as np

from danceapp import FEEDBACK_COLORS, PoseApp, UIState
from utils.pose_utils import Pose


def test_all_required_ui_states_exist() -> None:
    assert {state.value for state in UIState} == {
        "empty",
        "video_loaded",
        "analyzing",
        "ready",
        "running",
        "pose_lost",
        "camera_error",
        "finished",
    }


def test_time_format_is_stable() -> None:
    assert PoseApp._format_time(0) == "00:00"
    assert PoseApp._format_time(65.9) == "01:05"
    assert PoseApp._format_time(-5) == "00:00"


def test_pose_lost_gate_requires_reliable_torso(full_body_pose: Pose) -> None:
    assert PoseApp._is_scoreable_pose(full_body_pose)
    xy = np.full((17, 2), np.nan, dtype=np.float32)
    confidence = np.zeros(17, dtype=np.float32)
    xy[9] = [10.0, 10.0]
    xy[10] = [20.0, 10.0]
    confidence[9:11] = 0.9
    assert not PoseApp._is_scoreable_pose(Pose(xy, confidence))


def test_feedback_palette_covers_every_game_label() -> None:
    assert set(FEEDBACK_COLORS) == {"Perfect!", "Super!", "Good", "Miss"}
