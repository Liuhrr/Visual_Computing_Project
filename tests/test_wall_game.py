"""Unit tests for the Hole in the Wall bonus activity."""

from __future__ import annotations

import numpy as np
import pytest

from utils.pose_utils import Pose, empty_pose, valid_keypoints
from wall_game import FreezeGame, Wall, WallGame, WallState


def _sample_pose() -> Pose:
    xy = np.array(
        [
            [320.0, 120.0],
            [300.0, 110.0],
            [340.0, 110.0],
            [290.0, 130.0],
            [350.0, 130.0],
            [300.0, 200.0],
            [340.0, 200.0],
            [270.0, 280.0],
            [370.0, 280.0],
            [260.0, 360.0],
            [380.0, 360.0],
            [310.0, 400.0],
            [330.0, 400.0],
            [300.0, 520.0],
            [340.0, 520.0],
            [290.0, 640.0],
            [350.0, 640.0],
        ],
        dtype=np.float32,
    )
    confidence = np.ones(17, dtype=np.float32)
    return Pose(xy=xy, confidence=confidence)


def test_wall_spawns_and_judges_matching_pose() -> None:
    target = _sample_pose()
    wall = Wall(target, spawn_time=0.0)
    schedule = {"spawn_times": [0.0], "source": "fixed"}
    game = WallGame([{"xy": target.xy.tolist(), "confidence": target.confidence.tolist()}], schedule)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    elapsed = 0.0
    events = []
    while elapsed < 6.0:
        frame, ev = game.update(elapsed, target, frame)
        events.extend(ev)
        elapsed += 0.033

    assert any(e.kind == "perfect" for e in events)
    assert game.active_wall is None or game.active_wall.state == WallState.IDLE


def test_wall_judges_wrong_pose_as_miss() -> None:
    target = _sample_pose()
    wrong = target.xy.copy()
    wrong[[7, 8, 9, 10], 1] -= 150.0
    wrong_pose = Pose(xy=wrong, confidence=target.confidence.copy())

    schedule = {"spawn_times": [0.0], "source": "fixed"}
    game = WallGame([{"xy": target.xy.tolist(), "confidence": target.confidence.tolist()}], schedule)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    elapsed = 0.0
    events = []
    while elapsed < 6.0:
        frame, ev = game.update(elapsed, wrong_pose, frame)
        events.extend(ev)
        elapsed += 0.033

    assert any(e.kind == "miss" for e in events)


def test_freeze_detects_movement() -> None:
    target = _sample_pose()
    freeze = FreezeGame(events=[{"start": 0.0, "duration": 0.2}], enabled=True)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    event = None
    for t in np.arange(0.0, 0.5, 0.033):
        # Move a lot
        moved = target.xy.copy()
        moved[:, 0] += t * 300.0
        pose = Pose(xy=moved, confidence=target.confidence.copy())
        _, event = freeze.update(t, pose, frame.copy())
        if event is not None:
            break

    assert event is not None
    assert event.kind == "freeze_move"
    assert event.score < 0


def test_freeze_rewards_stillness() -> None:
    target = _sample_pose()
    freeze = FreezeGame(events=[{"start": 0.0, "duration": 0.2}], enabled=True)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    event = None
    for t in np.arange(0.0, 0.5, 0.033):
        # Hold almost still
        noise = np.random.normal(0.0, 0.5, target.xy.shape).astype(np.float32)
        pose = Pose(xy=target.xy + noise, confidence=target.confidence.copy())
        _, event = freeze.update(t, pose, frame.copy())
        if event is not None:
            break

    assert event is not None
    assert event.kind == "freeze_still"
    assert event.score > 0
