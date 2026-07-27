import random

import numpy as np

from silhouette_game import (
    ACTION_POSES,
    MATCH_THRESHOLD,
    best_pose_score,
    choose_next_action,
    pose_is_visible,
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
