from __future__ import annotations

import numpy as np
import pytest

from utils.pose_utils import Pose


@pytest.fixture
def full_body_pose() -> Pose:
    xy = np.array(
        [
            [0.0, -3.2],
            [-0.2, -3.3],
            [0.2, -3.3],
            [-0.45, -3.15],
            [0.45, -3.15],
            [-1.0, -2.0],
            [1.0, -2.0],
            [-2.0, -1.0],
            [2.0, -1.0],
            [-2.8, 0.2],
            [2.8, 0.2],
            [-0.8, 0.0],
            [0.8, 0.0],
            [-0.8, 2.0],
            [0.8, 2.0],
            [-0.8, 4.0],
            [0.8, 4.0],
        ],
        dtype=np.float32,
    )
    return Pose(xy, np.full(17, 0.95, dtype=np.float32))
