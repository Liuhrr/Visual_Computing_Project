"""Confidence-aware spatial and temporal dance scoring."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence
from collections import deque

import numpy as np

from .pose_utils import SKELETON, Pose, extract_angles, normalize_pose, valid_keypoints
ACTION_KEYPOINTS = [7, 8, 9, 10, 13, 14, 15, 16]


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
    common = np.array([valid_keypoints(ref)[i] and valid_keypoints(usr)[i] for i in ACTION_KEYPOINTS])
    count = int(np.count_nonzero(common))
    if count < 2:
        return None, count

    ref_xy = ref.xy[ACTION_KEYPOINTS][common]
    usr_xy = usr.xy[ACTION_KEYPOINTS][common]
    distance = np.linalg.norm(ref_xy - usr_xy, axis=1)
    point_similarity = np.exp(-np.square(distance / 0.75))
    point_weights = np.minimum(
        ref.confidence[ACTION_KEYPOINTS][common],
        usr.confidence[ACTION_KEYPOINTS][common]
    )
    point_score = _weighted_mean(point_similarity, point_weights)

    limb_values = []
    limb_weights = []
    action_connections = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]
    for start, end in action_connections:
        if (common[ACTION_KEYPOINTS.index(start)] if start in ACTION_KEYPOINTS else False) and \
           (common[ACTION_KEYPOINTS.index(end)] if end in ACTION_KEYPOINTS else False):
            if start in ACTION_KEYPOINTS and end in ACTION_KEYPOINTS:
                ref_vector = ref.xy[end] - ref.xy[start]
                user_vector = usr.xy[end] - usr.xy[start]
                denominator = float(np.linalg.norm(ref_vector) * np.linalg.norm(user_vector))
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
    if limb_values:
        limb_score = _weighted_mean(
            np.asarray(limb_values, dtype=np.float32),
            np.asarray(limb_weights, dtype=np.float32),
        )
    else:
        limb_score = None

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
    common = np.array([
        valid_keypoints(prev_ref)[i] and valid_keypoints(current_ref)[i] and
        valid_keypoints(prev_user)[i] and valid_keypoints(current_user)[i]
        for i in ACTION_KEYPOINTS
    ])
    if np.count_nonzero(common) < 2:
        return None

    ref_velocity = current_ref.xy[ACTION_KEYPOINTS][common] - prev_ref.xy[ACTION_KEYPOINTS][common]
    user_velocity = current_user.xy[ACTION_KEYPOINTS][common] - prev_user.xy[ACTION_KEYPOINTS][common]
    ref_speed = np.linalg.norm(ref_velocity, axis=1)
    user_speed = np.linalg.norm(user_velocity, axis=1)
    active = (ref_speed + user_speed) > 0.025
    if np.count_nonzero(active) < 2:
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
    angle = _angle_score(reference, user)
    position, compared = _spatial_score(reference, user)
    motion = _motion_score(previous_reference, reference, previous_user, user)
    body_common = np.array([
        valid_keypoints(reference)[i] and valid_keypoints(user)[i]
        for i in ACTION_KEYPOINTS
    ])
    coverage = float(np.count_nonzero(body_common) / len(ACTION_KEYPOINTS))

    components = [
        (angle, 0.35),
        (position, 0.25),
        (motion, 0.40),
    ]
    available = [(value, weight) for value, weight in components if value is not None]
    if not available or compared < 2:
        total = 0.0
    else:
        numerator = sum(float(value) * weight for value, weight in available)
        denominator = sum(weight for _, weight in available)
        base = numerator / denominator
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
    if score >= 0.92:
        return "Perfect!", (80, 240, 80)
    if score >= 0.82:
        return "Super!", (80, 220, 255)
    if score >= 0.68:
        return "Good", (40, 165, 255)
    return "Miss", (80, 80, 255)

class WindowScorer:
    def __init__(self, window_size: int = 15, punish_threshold: float = 0.35):
        self.window_size = window_size
        self.punish_threshold = punish_threshold
        self.ref_poses = deque(maxlen=window_size)
        self.user_poses = deque(maxlen=window_size)
        self._ready = False

    def add_frame(self, ref_pose: Pose, user_pose: Pose):
        self.ref_poses.append(ref_pose)
        self.user_poses.append(user_pose)
        if len(self.ref_poses) == self.window_size:
            self._ready = True

    def compute_window_score(self) -> Optional[ScoreBreakdown]:
        if not self._ready:
            return None

        frame_scores = []
        user_motion_amplitudes = []
        ref_motion_amplitudes = []
        user_small_motion_count = 0

        for i in range(self.window_size):
            ref = self.ref_poses[i]
            user = self.user_poses[i]
            prev_ref = self.ref_poses[i-1] if i > 0 else None
            prev_user = self.user_poses[i-1] if i > 0 else None
            breakdown = compute_pose_score(ref, user, prev_ref, prev_user)
            frame_scores.append(breakdown.total)

            if i > 0:
                user_center = user.center
                prev_user_center = prev_user.center if prev_user else user.center
                ref_center = ref.center
                prev_ref_center = prev_ref.center if prev_ref else ref.center

                user_motion = 0.0
                ref_motion = 0.0
                if np.isfinite(user_center).all() and np.isfinite(prev_user_center).all():
                    user_motion = float(np.linalg.norm(user_center - prev_user_center))
                if np.isfinite(ref_center).all() and np.isfinite(prev_ref_center).all():
                    ref_motion = float(np.linalg.norm(ref_center - prev_ref_center))

                user_motion_amplitudes.append(user_motion)
                ref_motion_amplitudes.append(ref_motion)

                if user_motion < 0.001:
                    user_small_motion_count += 1

        avg_score = float(np.mean(frame_scores))

        if user_motion_amplitudes and ref_motion_amplitudes:
            mean_user = float(np.mean(user_motion_amplitudes))
            mean_ref = float(np.mean(ref_motion_amplitudes))
            if mean_ref > 1e-6:
                ratio = mean_user / mean_ref
                if ratio < self.punish_threshold:
                    penalty = np.exp(-ratio * 20.0)
                    avg_score *= penalty

        if user_small_motion_count > self.window_size * 0.3:
            avg_score *= 0.2

        last_breakdown = compute_pose_score(
            self.ref_poses[-1], self.user_poses[-1],
            self.ref_poses[-2] if len(self.ref_poses) > 1 else None,
            self.user_poses[-2] if len(self.user_poses) > 1 else None
        )

        lags = []
        for i in range(self.window_size):
            ref = self.ref_poses[i]
            user = self.user_poses[i]
            prev_ref = self.ref_poses[i-1] if i > 0 else None
            prev_user = self.user_poses[i-1] if i > 0 else None
            bd = compute_pose_score(ref, user, prev_ref, prev_user)
            lags.append(bd.lag_ms)
        avg_lag = float(np.mean(lags)) if lags else 0.0

        return ScoreBreakdown(
            total=float(np.clip(avg_score, 0.0, 1.0)),
            angle=last_breakdown.angle,
            position=last_breakdown.position,
            motion=last_breakdown.motion,
            coverage=last_breakdown.coverage,
            compared_keypoints=last_breakdown.compared_keypoints,
            lag_ms=avg_lag
        )

    def reset(self):
        self.ref_poses.clear()
        self.user_poses.clear()
        self._ready = False