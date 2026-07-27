import random

import cv2
import numpy as np

from silhouette_game import (
    ACTION_POSES,
    CAMERA_PREVIEW_HEIGHT,
    CAMERA_PREVIEW_WIDTH,
    MATCH_THRESHOLD,
    best_pose_score,
    choose_next_action,
    fit_camera_frame,
    pose_is_visible,
    request_widest_camera_view,
)
from utils.pose_utils import Pose


def test_all_action_templates_have_full_body_points() -> None:
    assert len(ACTION_POSES) >= 6
    for action in ACTION_POSES:
        assert pose_is_visible(action.pose)
        assert np.isfinite(action.pose.xy[[5, 6, 11, 12]]).all()


def test_random_action_does_not_repeat_immediately() -> None:
    rng = random.Random(42)
    previous = 3
    for _ in range(50):
        current = choose_next_action(previous, rng)
        assert current != previous
        previous = current


def test_identical_action_clears_match_threshold() -> None:
    action = ACTION_POSES[0]
    assert best_pose_score(action.pose, action.pose) > MATCH_THRESHOLD


def test_missing_torso_is_not_scoreable() -> None:
    action = ACTION_POSES[0].pose
    xy = action.xy.copy()
    confidence = action.confidence.copy()
    xy[[5, 6, 11, 12]] = np.nan
    confidence[[5, 6, 11, 12]] = 0.0
    partial = Pose(xy=xy, confidence=confidence)
    assert not pose_is_visible(partial)
    assert best_pose_score(action, partial) == 0.0


def test_camera_preview_has_fixed_size_without_cropping() -> None:
    frame = np.full((360, 640, 3), 255, dtype=np.uint8)
    fitted = fit_camera_frame(frame)
    assert fitted.shape == (
        CAMERA_PREVIEW_HEIGHT,
        CAMERA_PREVIEW_WIDTH,
        3,
    )
    assert np.all(fitted[0] == 10)
    assert np.all(fitted[-1] == 10)
    assert np.all(fitted[CAMERA_PREVIEW_HEIGHT // 2] == 255)


def test_camera_requests_manual_minimum_zoom() -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self.zoom = 180.0
            self.requested = None

        def set(self, prop: int, value: float) -> bool:
            self.requested = (prop, value)
            self.zoom = 0.0
            return True

        def get(self, prop: int) -> float:
            assert prop == cv2.CAP_PROP_ZOOM
            return self.zoom

    capture = FakeCapture()
    locked, zoom = request_widest_camera_view(capture)
    assert locked
    assert zoom == 0.0
    assert capture.requested == (cv2.CAP_PROP_ZOOM, 0.0)


def test_unsupported_camera_zoom_is_reported() -> None:
    class UnsupportedCapture:
        def set(self, _prop: int, _value: float) -> bool:
            return False

        def get(self, _prop: int) -> float:
            return -1.0

    assert request_widest_camera_view(UnsupportedCapture()) == (False, -1.0)
