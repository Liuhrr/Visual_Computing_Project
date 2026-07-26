"""Confidence-aware spatial and temporal dance scoring."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np

from .pose_utils import SKELETON, Pose, extract_angles, normalize_pose, valid_keypoints


@dataclass(frozen=True)
class ScoreBreakdown:
    total: float
    angle: float
    position: float
    motion: Optional[float]
    coverage: float
    compared_keypoints: int
    lag_ms: float = 0.0


@dataclass(frozen=True)
class AlignmentResult:
    reference_index: int
    lag_seconds: float
    breakdown: ScoreBreakdown


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> Optional[float]:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return None
    return float(np.average(values[valid], weights=weights[valid]))


def _angle_score(reference: Pose, user: Pose) -> Optional[float]:
    ref_angles, ref_conf = extract_angles(reference)
    user_angles, user_conf = extract_angles(user)
    valid = np.isfinite(ref_angles) & np.isfinite(user_angles)
    if not np.any(valid):
        return None
    difference = np.abs(ref_angles - user_angles)
    # A 30-degree error is still recognizable but is no longer "Perfect".
    similarity = np.exp(-np.square(difference / 45.0))
    weights = np.minimum(ref_conf, user_conf)
    return _weighted_mean(similarity[valid], weights[valid])


def _spatial_score(reference: Pose, user: Pose) -> tuple[Optional[float], int]:
    ref = normalize_pose(reference)
    usr = normalize_pose(user)
    common = valid_keypoints(ref) & valid_keypoints(usr)
    count = int(np.count_nonzero(common))
    if count < 3:
        return None, count

    distance = np.linalg.norm(ref.xy[common] - usr.xy[common], axis=1)
    point_similarity = np.exp(-np.square(distance / 0.75))
    point_weights = np.minimum(ref.confidence[common], usr.confidence[common])
    point_score = _weighted_mean(point_similarity, point_weights)

    limb_values: list[float] = []
    limb_weights: list[float] = []
    for start, end in SKELETON:
        if not (common[start] and common[end]):
            continue
        ref_vector = ref.xy[end] - ref.xy[start]
        user_vector = usr.xy[end] - usr.xy[start]
        denominator = float(
            np.linalg.norm(ref_vector) * np.linalg.norm(user_vector)
        )
        if denominator <= 1e-8:
            continue
        cosine = float(np.clip(np.dot(ref_vector, user_vector) / denominator, -1, 1))
        limb_values.append((cosine + 1.0) / 2.0)
        limb_weights.append(
            min(
                ref.confidence[start],
                ref.confidence[end],
                usr.confidence[start],
                usr.confidence[end],
            )
        )
    limb_score = _weighted_mean(
        np.asarray(limb_values, dtype=np.float32),
        np.asarray(limb_weights, dtype=np.float32),
    )

    if point_score is None:
        return limb_score, count
    if limb_score is None:
        return point_score, count
    return 0.65 * point_score + 0.35 * limb_score, count


def _motion_score(
    previous_reference: Optional[Pose],
    reference: Pose,
    previous_user: Optional[Pose],
    user: Pose,
) -> Optional[float]:
    if previous_reference is None or previous_user is None:
        return None
    prev_ref = normalize_pose(previous_reference)
    current_ref = normalize_pose(reference)
    prev_user = normalize_pose(previous_user)
    current_user = normalize_pose(user)
    common = (
        valid_keypoints(prev_ref)
        & valid_keypoints(current_ref)
        & valid_keypoints(prev_user)
        & valid_keypoints(current_user)
    )
    if np.count_nonzero(common) < 3:
        return None

    ref_velocity = current_ref.xy[common] - prev_ref.xy[common]
    user_velocity = current_user.xy[common] - prev_user.xy[common]
    ref_speed = np.linalg.norm(ref_velocity, axis=1)
    user_speed = np.linalg.norm(user_velocity, axis=1)
    active = (ref_speed + user_speed) > 0.025
    if np.count_nonzero(active) < 2:
        # Holding the same pose is valid; do not punish it as a motion failure.
        return None

    speed_similarity = np.exp(
        -np.abs(ref_speed[active] - user_speed[active])
        / (ref_speed[active] + user_speed[active] + 0.08)
    )
    denominator = ref_speed[active] * user_speed[active]
    direction = np.zeros_like(denominator)
    moving_both = denominator > 1e-6
    direction[moving_both] = (
        np.sum(
            ref_velocity[active][moving_both]
            * user_velocity[active][moving_both],
            axis=1,
        )
        / denominator[moving_both]
        + 1.0
    ) / 2.0
    values = 0.55 * speed_similarity + 0.45 * np.clip(direction, 0.0, 1.0)
    return float(np.mean(values))


def compute_pose_score(
    reference: Pose,
    user: Pose,
    previous_reference: Optional[Pose] = None,
    previous_user: Optional[Pose] = None,
) -> ScoreBreakdown:
    """Score a user pose after confidence filtering and spatial normalization."""

    angle = _angle_score(reference, user)
    position, compared = _spatial_score(reference, user)
    motion = _motion_score(previous_reference, reference, previous_user, user)
    body_common = (
        valid_keypoints(reference)[5:17] & valid_keypoints(user)[5:17]
    )
    coverage = float(np.count_nonzero(body_common) / len(body_common))

    components: list[tuple[Optional[float], float]] = [
        (angle, 0.50),
        (position, 0.35),
        (motion, 0.15),
    ]
    available = [(value, weight) for value, weight in components if value is not None]
    if not available or compared < 3:
        total = 0.0
    else:
        numerator = sum(float(value) * weight for value, weight in available)
        denominator = sum(weight for _, weight in available)
        base = numerator / denominator
        # Limited partial-body poses remain scoreable, but full coverage is rewarded.
        reliability = 0.72 + 0.28 * coverage
        total = float(np.clip(base * reliability, 0.0, 1.0))

    return ScoreBreakdown(
        total=total,
        angle=float(angle or 0.0),
        position=float(position or 0.0),
        motion=motion,
        coverage=coverage,
        compared_keypoints=compared,
    )


class TemporalAligner:
    """Find the best reference pose near the current playback time.

    The local search handles ordinary webcam/reaction lag without allowing an
    unrelated pose from a distant part of the dance to win.
    """

    def __init__(
        self,
        reference_poses: Sequence[Pose],
        timestamps: Sequence[float],
        search_window_seconds: float = 0.45,
    ) -> None:
        if len(reference_poses) != len(timestamps):
            raise ValueError("Reference poses and timestamps must have equal length")
        if not reference_poses:
            raise ValueError("Reference sequence cannot be empty")
        self.reference_poses = reference_poses
        self.timestamps = np.asarray(timestamps, dtype=np.float64)
        self.search_window_seconds = float(search_window_seconds)
        self._smoothed_lag_seconds = 0.0

    def reset(self) -> None:
        self._smoothed_lag_seconds = 0.0

    def align(
        self,
        user_pose: Pose,
        playback_seconds: float,
        previous_user_pose: Optional[Pose] = None,
    ) -> AlignmentResult:
        center = int(np.searchsorted(self.timestamps, playback_seconds))
        center = int(np.clip(center, 0, len(self.timestamps) - 1))
        start_time = playback_seconds - self.search_window_seconds
        end_time = playback_seconds + self.search_window_seconds
        start = int(max(0, np.searchsorted(self.timestamps, start_time, side="left")))
        end = int(
            min(
                len(self.timestamps),
                np.searchsorted(self.timestamps, end_time, side="right"),
            )
        )
        if end <= start:
            start, end = center, center + 1

        best_index = center
        best_breakdown = compute_pose_score(
            self.reference_poses[center], user_pose, previous_user=previous_user_pose
        )
        best_objective = -1.0
        for index in range(start, end):
            previous_reference = (
                self.reference_poses[index - 1] if index > 0 else None
            )
            breakdown = compute_pose_score(
                self.reference_poses[index],
                user_pose,
                previous_reference,
                previous_user_pose,
            )
            temporal_distance = abs(self.timestamps[index] - playback_seconds)
            penalty = 0.035 * temporal_distance / max(
                self.search_window_seconds, 1e-6
            )
            objective = breakdown.total - penalty
            if objective > best_objective:
                best_objective = objective
                best_index = index
                best_breakdown = breakdown

        raw_lag = float(self.timestamps[best_index] - playback_seconds)
        self._smoothed_lag_seconds = (
            0.72 * self._smoothed_lag_seconds + 0.28 * raw_lag
        )
        breakdown = replace(
            best_breakdown, lag_ms=self._smoothed_lag_seconds * 1000.0
        )
        return AlignmentResult(
            reference_index=best_index,
            lag_seconds=self._smoothed_lag_seconds,
            breakdown=breakdown,
        )


def compute_angle_similarity(angles_ref, angles_user) -> float:
    """Backward-compatible angle score that ignores missing (NaN) angles."""

    reference = np.asarray(angles_ref, dtype=np.float32)
    user = np.asarray(angles_user, dtype=np.float32)
    if reference.shape != user.shape:
        return 0.0
    valid = np.isfinite(reference) & np.isfinite(user)
    if not np.any(valid):
        return 0.0
    difference = np.abs(reference[valid] - user[valid])
    return float(np.mean(np.exp(-np.square(difference / 45.0))))


def map_score_to_feedback(score: float) -> tuple[str, tuple[int, int, int]]:
    """Return feedback and an OpenCV BGR color."""

    if score >= 0.90:
        return "Perfect!", (80, 240, 80)
    if score >= 0.75:
        return "Super!", (80, 220, 255)
    if score >= 0.60:
        return "Good", (40, 165, 255)
    return "Miss", (80, 80, 255)
