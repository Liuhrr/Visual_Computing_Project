from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from utils.pose_utils import (
    Pose,
    PoseSmoother,
    mirror_pose,
    normalize_pose,
    select_target_person,
    valid_keypoints,
)


def test_normalization_is_translation_and_scale_invariant(full_body_pose: Pose) -> None:
    transformed = Pose(
        full_body_pose.xy * 73.0 + np.array([510.0, 280.0]),
        full_body_pose.confidence,
    )
    expected = normalize_pose(full_body_pose)
    actual = normalize_pose(transformed)
    common = valid_keypoints(expected) & valid_keypoints(actual)
    np.testing.assert_allclose(expected.xy[common], actual.xy[common], atol=1e-5)


def test_mirroring_twice_restores_pose(full_body_pose: Pose) -> None:
    restored = mirror_pose(mirror_pose(full_body_pose))
    np.testing.assert_allclose(restored.xy, full_body_pose.xy)
    np.testing.assert_allclose(restored.confidence, full_body_pose.confidence)


def test_smoother_recovers_one_short_detection_gap(full_body_pose: Pose) -> None:
    smoother = PoseSmoother(alpha=0.5, gap_decay=0.6)
    smoother.update(full_body_pose)
    with_gap = full_body_pose.xy.copy()
    confidence = full_body_pose.confidence.copy()
    with_gap[9] = np.nan
    confidence[9] = 0.0
    recovered = smoother.update(Pose(with_gap, confidence))
    assert recovered is not None
    assert np.isfinite(recovered.xy[9]).all()
    assert recovered.confidence[9] > 0.25


def test_target_selector_prefers_large_visible_central_dancer() -> None:
    small = np.full((17, 2), [80.0, 80.0], dtype=np.float32)
    large = np.column_stack(
        [np.linspace(230, 410, 17), np.linspace(80, 430, 17)]
    ).astype(np.float32)
    keypoints = SimpleNamespace(
        xy=np.stack([small, large]),
        conf=np.stack(
            [
                np.full(17, 0.45, dtype=np.float32),
                np.full(17, 0.92, dtype=np.float32),
            ]
        ),
    )
    boxes = SimpleNamespace(
        xyxy=np.array([[60, 60, 110, 150], [210, 50, 430, 460]], np.float32),
        conf=np.array([0.55, 0.96], np.float32),
        id=None,
    )
    result = SimpleNamespace(keypoints=keypoints, boxes=boxes)
    selection = select_target_person(result, (480, 640, 3))
    assert selection is not None
    assert selection.index == 1
