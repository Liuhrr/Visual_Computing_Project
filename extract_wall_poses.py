"""Offline extraction of iconic poses from a reference dance video.

The wall poses are not hand-authored: they are harvested from the reference
video by looking for short intervals where the dancer holds a stable shape
(low local motion) and selecting the most confident frame in each interval.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Make utils importable when running as script
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.pose_utils import (  # noqa: E402
    Pose,
    extract_angles,
    normalize_pose,
    valid_keypoints,
)
from utils.reference import ReferenceSequence, analyze_reference_video  # noqa: E402
from utils.scoring import compute_pose_score  # noqa: E402

import wall_config as cfg  # noqa: E402


def _pose_motion_energy(pose: Pose, prev_pose: Optional[Pose]) -> float:
    """Return a scalar motion energy for this frame.

    Combines normalized-keypoint displacement and joint-angle displacement.
    Missing keypoints are ignored. Higher = more movement.
    """
    if prev_pose is None:
        return 1.0

    norm_curr = normalize_pose(pose)
    norm_prev = normalize_pose(prev_pose)
    common = valid_keypoints(norm_curr) & valid_keypoints(norm_prev)
    if np.count_nonzero(common) < 3:
        return 1.0

    disp = np.linalg.norm(norm_curr.xy[common] - norm_prev.xy[common], axis=1)
    point_energy = float(np.mean(disp))

    curr_angles, _ = extract_angles(pose)
    prev_angles, _ = extract_angles(prev_pose)
    valid_angles = np.isfinite(curr_angles) & np.isfinite(prev_angles)
    if np.count_nonzero(valid_angles) >= 2:
        angle_disp = np.abs(curr_angles[valid_angles] - prev_angles[valid_angles])
        angle_energy = float(np.mean(angle_disp)) / 180.0
    else:
        angle_energy = 0.0

    return float(point_energy + 0.5 * angle_energy)


def _average_confidence(pose: Pose) -> float:
    body = valid_keypoints(pose)[5:17]
    if not np.any(body):
        return 0.0
    return float(np.mean(pose.confidence[5:17][body]))


def _visible_body_count(pose: Pose) -> int:
    return int(np.count_nonzero(valid_keypoints(pose)[5:17]))


def _is_valid_target_pose(pose: Pose) -> bool:
    return (
        _visible_body_count(pose) >= cfg.MIN_BODY_KEYPOINTS
        and _average_confidence(pose) >= cfg.MIN_POSE_CONFIDENCE
    )


def _pose_similarity(a: Pose, b: Pose) -> float:
    breakdown = compute_pose_score(a, b)
    return breakdown.total


def _find_still_intervals(
    energies: np.ndarray,
    stillness_percentile: float = cfg.STILLNESS_PERCENTILE,
    min_gap_seconds: float = 0.4,
    fps: float = 30.0,
) -> list[tuple[int, int]]:
    """Find low-motion intervals that correspond to held poses.

    Returns list of (start_index, end_index) inclusive intervals.
    """
    if len(energies) < 3:
        return []

    threshold = float(np.percentile(energies[1:], stillness_percentile))
    low = energies < threshold
    min_gap_frames = max(1, int(round(min_gap_seconds * fps)))

    intervals: list[tuple[int, int]] = []
    start: Optional[int] = None
    for i, is_low in enumerate(low):
        if is_low and start is None:
            start = i
        elif not is_low and start is not None:
            if i - start >= min_gap_frames:
                intervals.append((start, i - 1))
            start = None
    if start is not None and len(energies) - start >= min_gap_frames:
        intervals.append((start, len(energies) - 1))
    return intervals


def extract_wall_poses(
    sequence: ReferenceSequence,
    min_count: int = cfg.MIN_POSE_COUNT,
    max_count: int = cfg.MAX_POSE_COUNT,
    dedup_threshold: float = cfg.SIMILARITY_DEDUP_THRESHOLD,
) -> list[dict]:
    """Select a small set of iconic wall poses from a reference sequence."""
    poses = sequence.poses
    timestamps = sequence.timestamps
    fps = sequence.fps

    # 1) compute per-frame motion energy
    energies = np.zeros(len(poses), dtype=np.float32)
    prev: Optional[Pose] = None
    for i, pose in enumerate(poses):
        energies[i] = _pose_motion_energy(pose, prev)
        prev = pose

    # 2) find low-motion intervals
    intervals = _find_still_intervals(energies, fps=fps)

    # 3) pick best frame per interval
    candidates: list[tuple[int, Pose]] = []
    for start, end in intervals:
        best_idx = start
        best_conf = -1.0
        for idx in range(start, end + 1):
            if not _is_valid_target_pose(poses[idx]):
                continue
            conf = _average_confidence(poses[idx])
            if conf > best_conf:
                best_conf = conf
                best_idx = idx
        if best_conf > 0.0:
            candidates.append((best_idx, poses[best_idx]))

    # 4) fallback: if too few, sample uniformly across valid frames
    if len(candidates) < min_count:
        valid_indices = [
            i for i, pose in enumerate(poses) if _is_valid_target_pose(pose)
        ]
        step = max(1, len(valid_indices) // min_count)
        extra = [
            (i, poses[i])
            for i in valid_indices[::step][: min_count - len(candidates)]
        ]
        candidates.extend(extra)

    candidates.sort(key=lambda item: item[0])

    # 5) deduplicate by similarity
    selected: list[tuple[int, Pose]] = []
    for idx, pose in candidates:
        if len(selected) >= max_count:
            break
        if not selected:
            selected.append((idx, pose))
            continue
        last_idx, last_pose = selected[-1]
        if _pose_similarity(last_pose, pose) >= dedup_threshold:
            # keep the more confident one
            if _average_confidence(pose) > _average_confidence(last_pose):
                selected[-1] = (idx, pose)
        else:
            selected.append((idx, pose))

    # 6) final cap
    selected = selected[:max_count]

    result = []
    for idx, pose in selected:
        result.append(
            {
                "timestamp": float(timestamps[idx]),
                "frame_index": int(idx),
                "xy": pose.xy.tolist(),
                "confidence": pose.confidence.tolist(),
                "source_video": str(sequence.source_path),
            }
        )
    return result


def save_wall_poses(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": 1,
                "count": len(data),
                "poses": data,
            },
            f,
            indent=2,
        )


def load_wall_poses(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("poses", [])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract iconic wall poses from a reference dance video."
    )
    parser.add_argument("video", type=Path, help="Path to the reference dance video")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "yolov8n-pose.pt",
        help="Path to the YOLOv8-pose model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=cfg.DEFAULT_WALL_POSES_PATH,
        help="Output JSON path",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-analyze even if a cached reference exists",
    )
    args = parser.parse_args()

    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")

    cache_dir = PROJECT_ROOT / "cache" / "reference"
    sequence = analyze_reference_video(
        args.video,
        args.model,
        cache_dir,
        force=args.force,
        progress=lambda cur, total: print(f"Analyzing reference: {cur}/{total}"),
    )

    print(f"Reference: {sequence.frame_count} frames @ {sequence.fps:.2f} FPS")
    poses = extract_wall_poses(sequence)
    save_wall_poses(poses, args.output)
    print(f"Saved {len(poses)} wall poses to {args.output}")


if __name__ == "__main__":
    main()
