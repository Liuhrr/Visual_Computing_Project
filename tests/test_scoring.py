from __future__ import annotations

import numpy as np

from utils.pose_utils import Pose, mirror_pose
from utils.scoring import TemporalAligner, compute_pose_score, map_score_to_feedback


def test_identical_pose_scores_near_perfect(full_body_pose: Pose) -> None:
    score = compute_pose_score(full_body_pose, full_body_pose)
    assert score.total > 0.98
    assert score.coverage == 1.0


def test_score_is_translation_and_scale_invariant(full_body_pose: Pose) -> None:
    camera_pose = Pose(
        full_body_pose.xy * 91.0 + np.array([640.0, 360.0]),
        full_body_pose.confidence,
    )
    score = compute_pose_score(full_body_pose, camera_pose)
    assert score.total > 0.98


def test_wrong_limb_configuration_scores_lower(full_body_pose: Pose) -> None:
    wrong_xy = full_body_pose.xy.copy()
    wrong_xy[7] = [-0.5, -3.0]
    wrong_xy[9] = [0.8, -3.7]
    wrong_xy[8] = [0.5, -3.0]
    wrong_xy[10] = [-0.8, -3.7]
    wrong = Pose(wrong_xy, full_body_pose.confidence)
    correct_score = compute_pose_score(full_body_pose, full_body_pose).total
    wrong_score = compute_pose_score(full_body_pose, wrong).total
    assert wrong_score < correct_score - 0.18


def test_partial_body_is_scoreable_but_reports_coverage(full_body_pose: Pose) -> None:
    xy = full_body_pose.xy.copy()
    confidence = full_body_pose.confidence.copy()
    xy[13:17] = np.nan
    confidence[13:17] = 0.0
    partial = Pose(xy, confidence)
    score = compute_pose_score(full_body_pose, partial)
    assert score.total > 0.65
    assert 0.5 < score.coverage < 0.8


def test_mirror_matching_recovers_asymmetric_pose(full_body_pose: Pose) -> None:
    asymmetric_xy = full_body_pose.xy.copy()
    asymmetric_xy[7] = [-1.0, -3.0]
    asymmetric_xy[9] = [-0.8, -4.2]
    reference = Pose(asymmetric_xy, full_body_pose.confidence)
    mirrored_player = mirror_pose(reference)
    raw_score = compute_pose_score(reference, mirrored_player).total
    matched_score = compute_pose_score(
        reference, mirror_pose(mirrored_player)
    ).total
    assert matched_score > 0.98
    assert matched_score > raw_score + 0.10


def test_temporal_alignment_finds_nearby_matching_frame(
    full_body_pose: Pose,
) -> None:
    poses = []
    for index in range(21):
        xy = full_body_pose.xy.copy()
        angle = -2.7 + index * 0.14
        xy[9] = xy[7] + np.array([np.cos(angle), np.sin(angle)]) * 2.0
        poses.append(Pose(xy, full_body_pose.confidence))
    timestamps = np.arange(21, dtype=np.float64) * 0.1
    aligner = TemporalAligner(poses, timestamps, search_window_seconds=0.45)
    result = aligner.align(poses[12], playback_seconds=1.0)
    assert abs(result.reference_index - 12) <= 1
    assert result.lag_seconds > 0.0


def test_feedback_boundaries() -> None:
    assert map_score_to_feedback(0.90)[0] == "Perfect!"
    assert map_score_to_feedback(0.75)[0] == "Super!"
    assert map_score_to_feedback(0.60)[0] == "Good"
    assert map_score_to_feedback(0.59)[0] == "Miss"
