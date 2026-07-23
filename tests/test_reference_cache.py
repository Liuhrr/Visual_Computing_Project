from __future__ import annotations

import numpy as np

from utils.pose_utils import Pose
from utils.reference import ReferenceSequence, load_reference, save_reference


def test_reference_cache_round_trip(tmp_path, full_body_pose: Pose) -> None:
    sequence = ReferenceSequence(
        poses=[full_body_pose, full_body_pose],
        timestamps=np.array([0.0, 1.0 / 30.0]),
        fps=30.0,
        frame_count=2,
        width=640,
        height=480,
        source_path="/tmp/reference.mp4",
    )
    destination = tmp_path / "reference.npz"
    save_reference(sequence, destination)
    loaded = load_reference(destination)
    assert loaded.from_cache
    assert loaded.frame_count == 2
    assert loaded.fps == 30.0
    np.testing.assert_allclose(loaded.poses[0].xy, full_body_pose.xy)
