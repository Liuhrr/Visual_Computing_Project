"""Pose extraction, target selection, normalization, smoothing, and drawing.

The project uses the 17-keypoint COCO convention returned by YOLOv8-pose.
Missing keypoints are represented by ``NaN`` coordinates instead of ``(0, 0)``;
this prevents invisible joints from accidentally affecting the score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import numpy as np


KEYPOINT_NAMES = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

SKELETON = (
    (0, 5),
    (0, 6),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

# (point A, angle vertex B, point C)
ANGLE_TRIPLETS = (
    (5, 7, 9),    # left elbow
    (6, 8, 10),   # right elbow
    (11, 13, 15), # left knee
    (12, 14, 16), # right knee
)

LEFT_RIGHT_PAIRS = (
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
)


@dataclass(frozen=True)
class Pose:
    """A single person's pose in image coordinates."""

    xy: np.ndarray
    confidence: np.ndarray
    bbox: Optional[np.ndarray] = None
    track_id: Optional[int] = None

    def __post_init__(self) -> None:
        xy = np.asarray(self.xy, dtype=np.float32)
        confidence = np.asarray(self.confidence, dtype=np.float32)
        if xy.shape != (17, 2):
            raise ValueError(f"Expected xy shape (17, 2), got {xy.shape}")
        if confidence.shape != (17,):
            raise ValueError(
                f"Expected confidence shape (17,), got {confidence.shape}"
            )
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "confidence", np.clip(confidence, 0.0, 1.0))
        if self.bbox is not None:
            object.__setattr__(
                self, "bbox", np.asarray(self.bbox, dtype=np.float32).reshape(4)
            )

    @property
    def center(self) -> np.ndarray:
        valid = valid_keypoints(self)
        if not np.any(valid):
            return np.array([np.nan, np.nan], dtype=np.float32)
        return np.nanmean(self.xy[valid], axis=0).astype(np.float32)


@dataclass(frozen=True)
class TargetSelection:
    index: int
    track_id: Optional[int]
    score: float


def empty_pose() -> Pose:
    return Pose(
        np.full((17, 2), np.nan, dtype=np.float32),
        np.zeros(17, dtype=np.float32),
    )


def valid_keypoints(pose: Pose, threshold: float = 0.25) -> np.ndarray:
    return (
        np.isfinite(pose.xy).all(axis=1)
        & np.isfinite(pose.confidence)
        & (pose.confidence >= threshold)
    )


def _as_numpy(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def select_target_person(
    result,
    frame_shape: tuple[int, ...],
    previous_center: Optional[np.ndarray] = None,
    preferred_track_id: Optional[int] = None,
    conf_threshold: float = 0.25,
) -> Optional[TargetSelection]:
    """Choose the likely dancer when several people are detected.

    Candidates are ranked by visible-keypoint coverage, bounding-box size,
    detection confidence, centrality, and temporal continuity. A persistent
    track ID receives a strong bonus when one is available.
    """

    if result is None or getattr(result, "keypoints", None) is None:
        return None
    all_xy = _as_numpy(result.keypoints.xy)
    all_conf = _as_numpy(result.keypoints.conf)
    if all_xy is None or len(all_xy) == 0:
        return None
    if all_conf is None:
        all_conf = np.ones(all_xy.shape[:2], dtype=np.float32)

    height, width = frame_shape[:2]
    diagonal = max(float(np.hypot(width, height)), 1.0)
    frame_area = max(float(width * height), 1.0)
    frame_center = np.array([width / 2.0, height / 2.0], dtype=np.float32)

    boxes = getattr(result, "boxes", None)
    box_xyxy = _as_numpy(getattr(boxes, "xyxy", None))
    box_conf = _as_numpy(getattr(boxes, "conf", None))
    box_ids = _as_numpy(getattr(boxes, "id", None))

    best: Optional[TargetSelection] = None
    for index, (xy, confidence) in enumerate(zip(all_xy, all_conf)):
        visible = np.isfinite(xy).all(axis=1) & (confidence >= conf_threshold)
        coverage = float(np.mean(visible))
        if not np.any(visible):
            continue
        center = np.mean(xy[visible], axis=0)

        if box_xyxy is not None and index < len(box_xyxy):
            x1, y1, x2, y2 = box_xyxy[index]
            area = max(float((x2 - x1) * (y2 - y1)), 0.0) / frame_area
        else:
            visible_xy = xy[visible]
            span = np.ptp(visible_xy, axis=0)
            area = float(span[0] * span[1]) / frame_area
        area_score = min(area / 0.35, 1.0)

        detection_score = (
            float(box_conf[index])
            if box_conf is not None and index < len(box_conf)
            else float(np.mean(confidence[visible]))
        )
        centrality = 1.0 - min(float(np.linalg.norm(center - frame_center)) / diagonal, 1.0)

        continuity = 0.0
        if previous_center is not None and np.isfinite(previous_center).all():
            continuity = 1.0 - min(
                float(np.linalg.norm(center - previous_center)) / (0.35 * diagonal),
                1.0,
            )

        track_id = None
        if box_ids is not None and index < len(box_ids) and np.isfinite(box_ids[index]):
            track_id = int(box_ids[index])
        track_bonus = 0.35 if track_id is not None and track_id == preferred_track_id else 0.0

        score = (
            0.34 * coverage
            + 0.22 * area_score
            + 0.14 * detection_score
            + 0.10 * centrality
            + 0.20 * continuity
            + track_bonus
        )
        if best is None or score > best.score:
            best = TargetSelection(index=index, track_id=track_id, score=score)
    return best


def pose_from_result(
    result,
    selection: Optional[TargetSelection] = None,
    conf_threshold: float = 0.25,
) -> Optional[Pose]:
    if result is None or getattr(result, "keypoints", None) is None:
        return None
    all_xy = _as_numpy(result.keypoints.xy)
    all_conf = _as_numpy(result.keypoints.conf)
    if all_xy is None or len(all_xy) == 0:
        return None
    index = selection.index if selection is not None else 0
    if index >= len(all_xy):
        return None
    confidence = (
        all_conf[index].astype(np.float32)
        if all_conf is not None
        else np.ones(17, dtype=np.float32)
    )
    xy = all_xy[index].astype(np.float32)
    invisible = (~np.isfinite(xy).all(axis=1)) | (confidence < conf_threshold)
    xy[invisible] = np.nan

    boxes = getattr(result, "boxes", None)
    all_boxes = _as_numpy(getattr(boxes, "xyxy", None))
    bbox = all_boxes[index] if all_boxes is not None and index < len(all_boxes) else None
    track_id = selection.track_id if selection is not None else None
    return Pose(xy=xy, confidence=confidence, bbox=bbox, track_id=track_id)


def extract_keypoints_from_result(
    result,
    width: Optional[int] = None,
    height: Optional[int] = None,
    conf_threshold: float = 0.25,
):
    """Backward-compatible wrapper returning ``[(x, y), ...]``."""

    del width, height
    pose = pose_from_result(result, conf_threshold=conf_threshold)
    if pose is None:
        return None
    return [
        (float(x), float(y)) if np.isfinite([x, y]).all() else (None, None)
        for x, y in pose.xy
    ]


class PoseSmoother:
    """Confidence-aware exponential smoothing with short-gap recovery."""

    def __init__(self, alpha: float = 0.65, gap_decay: float = 0.55) -> None:
        self.alpha = float(alpha)
        self.gap_decay = float(gap_decay)
        self._pose: Optional[Pose] = None

    def reset(self) -> None:
        self._pose = None

    def update(self, pose: Optional[Pose]) -> Optional[Pose]:
        if pose is None:
            if self._pose is None:
                return None
            confidence = self._pose.confidence * self.gap_decay
            xy = self._pose.xy.copy()
            xy[confidence < 0.25] = np.nan
            self._pose = Pose(xy, confidence, self._pose.bbox, self._pose.track_id)
            return self._pose
        if self._pose is None:
            self._pose = pose
            return pose

        current_valid = valid_keypoints(pose, 0.05)
        previous_valid = valid_keypoints(self._pose, 0.05)
        both = current_valid & previous_valid
        xy = pose.xy.copy()
        xy[both] = (
            self.alpha * pose.xy[both]
            + (1.0 - self.alpha) * self._pose.xy[both]
        )
        recovered = ~current_valid & previous_valid
        xy[recovered] = self._pose.xy[recovered]
        confidence = pose.confidence.copy()
        confidence[recovered] = self._pose.confidence[recovered] * self.gap_decay
        xy[confidence < 0.25] = np.nan
        self._pose = Pose(xy, confidence, pose.bbox, pose.track_id)
        return self._pose


def mirror_pose(pose: Pose) -> Pose:
    """Mirror normalized coordinates and swap anatomical left/right labels."""

    xy = pose.xy.copy()
    confidence = pose.confidence.copy()
    xy[:, 0] *= -1.0
    for left, right in LEFT_RIGHT_PAIRS:
        xy[[left, right]] = xy[[right, left]]
        confidence[[left, right]] = confidence[[right, left]]
    return Pose(xy, confidence, pose.bbox, pose.track_id)


def normalize_pose(pose: Pose, conf_threshold: float = 0.25) -> Pose:
    """Remove translation and body scale while preserving body shape."""

    valid = valid_keypoints(pose, conf_threshold)
    if np.count_nonzero(valid) < 3:
        return empty_pose()

    def midpoint(a: int, b: int) -> Optional[np.ndarray]:
        if valid[a] and valid[b]:
            return (pose.xy[a] + pose.xy[b]) / 2.0
        return None

    hip_mid = midpoint(11, 12)
    shoulder_mid = midpoint(5, 6)
    center = hip_mid
    if center is None:
        center = shoulder_mid
    if center is None:
        center = np.mean(pose.xy[valid], axis=0)

    scale_candidates: list[float] = []
    if hip_mid is not None and shoulder_mid is not None:
        scale_candidates.append(float(np.linalg.norm(hip_mid - shoulder_mid)))
    if valid[5] and valid[6]:
        scale_candidates.append(float(np.linalg.norm(pose.xy[5] - pose.xy[6])))
    if valid[11] and valid[12]:
        scale_candidates.append(float(np.linalg.norm(pose.xy[11] - pose.xy[12])))
    scale_candidates = [value for value in scale_candidates if value > 1e-4]
    if scale_candidates:
        scale = max(scale_candidates)
    else:
        span = np.ptp(pose.xy[valid], axis=0)
        scale = float(np.linalg.norm(span))
    if scale <= 1e-4:
        return empty_pose()

    normalized = (pose.xy - center) / scale
    normalized[~valid] = np.nan
    return Pose(normalized, pose.confidence.copy(), track_id=pose.track_id)


def calculate_angle(a, b, c) -> Optional[float]:
    points = np.asarray([a, b, c], dtype=np.float32)
    if not np.isfinite(points).all():
        return None
    ba = points[0] - points[1]
    bc = points[2] - points[1]
    denominator = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denominator <= 1e-8:
        return None
    cosine = float(np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def extract_angles(
    pose: Pose, conf_threshold: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    angles = np.full(len(ANGLE_TRIPLETS), np.nan, dtype=np.float32)
    confidence = np.zeros(len(ANGLE_TRIPLETS), dtype=np.float32)
    valid = valid_keypoints(pose, conf_threshold)
    for index, (a, b, c) in enumerate(ANGLE_TRIPLETS):
        if valid[a] and valid[b] and valid[c]:
            angle = calculate_angle(pose.xy[a], pose.xy[b], pose.xy[c])
            if angle is not None:
                angles[index] = angle
                confidence[index] = min(
                    pose.confidence[a], pose.confidence[b], pose.confidence[c]
                )
    return angles, confidence


def extract_angle_vector(keypoints) -> np.ndarray:
    """Backward-compatible angle extraction; missing angles remain ``NaN``."""

    xy = np.full((17, 2), np.nan, dtype=np.float32)
    confidence = np.zeros(17, dtype=np.float32)
    for index, point in enumerate(keypoints[:17]):
        if point is not None and None not in point:
            xy[index] = point
            confidence[index] = 1.0
    return extract_angles(Pose(xy, confidence))[0]


def draw_pose(
    frame: np.ndarray,
    pose: Optional[Pose],
    line_color: tuple[int, int, int] = (80, 220, 80),
    point_color: tuple[int, int, int] = (40, 80, 255),
    conf_threshold: float = 0.25,
    label: Optional[str] = None,
) -> np.ndarray:
    overlay = frame.copy()
    if pose is None:
        return overlay
    valid = valid_keypoints(pose, conf_threshold)
    for start, end in SKELETON:
        if valid[start] and valid[end]:
            p1 = tuple(np.rint(pose.xy[start]).astype(int))
            p2 = tuple(np.rint(pose.xy[end]).astype(int))
            cv2.line(overlay, p1, p2, line_color, 3, cv2.LINE_AA)
    for index in np.flatnonzero(valid):
        point = tuple(np.rint(pose.xy[index]).astype(int))
        cv2.circle(overlay, point, 5, point_color, -1, cv2.LINE_AA)
    if label and pose.bbox is not None:
        x1, y1 = np.rint(pose.bbox[:2]).astype(int)
        cv2.putText(
            overlay,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            line_color,
            2,
            cv2.LINE_AA,
        )
    return overlay


def poses_from_arrays(
    xy_sequence: Iterable[np.ndarray], confidence_sequence: Iterable[np.ndarray]
) -> list[Pose]:
    return [
        Pose(xy, confidence)
        for xy, confidence in zip(xy_sequence, confidence_sequence)
    ]
