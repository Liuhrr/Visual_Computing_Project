"""Reference-video analysis and pose-cache management."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from threading import Event
from typing import Callable, Optional

import cv2
import numpy as np

from .pose_utils import (
    Pose,
    PoseSmoother,
    pose_from_result,
    poses_from_arrays,
    select_target_person,
)


ProgressCallback = Callable[[int, int], None]


class ReferenceAnalysisCancelled(RuntimeError):
    pass


@dataclass
class ReferenceSequence:
    poses: list[Pose]
    timestamps: np.ndarray
    fps: float
    frame_count: int
    width: int
    height: int
    source_path: str
    from_cache: bool = False

    @property
    def duration(self) -> float:
        if self.frame_count <= 1:
            return 0.0
        return float(self.timestamps[-1])


def _cache_key(video_path: Path, model_path: Path) -> str:
    stat = video_path.stat()
    identity = (
        f"{video_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:"
        f"{model_path.name}:pose-cache-v3"
    )
    return sha1(identity.encode("utf-8")).hexdigest()[:12]


def cache_path_for(
    video_path: str | Path, model_path: str | Path, cache_dir: str | Path
) -> Path:
    video = Path(video_path)
    return Path(cache_dir) / f"{video.stem}-{_cache_key(video, Path(model_path))}.npz"


def save_reference(sequence: ReferenceSequence, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    xy = np.stack([pose.xy for pose in sequence.poses]).astype(np.float32)
    confidence = np.stack(
        [pose.confidence for pose in sequence.poses]
    ).astype(np.float32)
    np.savez_compressed(
        destination,
        xy=xy,
        confidence=confidence,
        timestamps=sequence.timestamps.astype(np.float64),
        fps=np.asarray(sequence.fps, dtype=np.float64),
        frame_count=np.asarray(sequence.frame_count, dtype=np.int64),
        width=np.asarray(sequence.width, dtype=np.int64),
        height=np.asarray(sequence.height, dtype=np.int64),
        source_path=np.asarray(sequence.source_path),
    )


def load_reference(cache_path: Path) -> ReferenceSequence:
    with np.load(cache_path, allow_pickle=False) as data:
        poses = poses_from_arrays(data["xy"], data["confidence"])
        return ReferenceSequence(
            poses=poses,
            timestamps=data["timestamps"].astype(np.float64),
            fps=float(data["fps"]),
            frame_count=int(data["frame_count"]),
            width=int(data["width"]),
            height=int(data["height"]),
            source_path=str(data["source_path"]),
            from_cache=True,
        )


def analyze_reference_video(
    video_path: str | Path,
    model_path: str | Path,
    cache_dir: str | Path,
    *,
    force: bool = False,
    progress: Optional[ProgressCallback] = None,
    cancel_event: Optional[Event] = None,
) -> ReferenceSequence:
    """Extract and cache the primary dancer pose for every reference frame."""

    video = Path(video_path).expanduser().resolve()
    model_file = Path(model_path).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Reference video not found: {video}")
    if not model_file.is_file():
        raise FileNotFoundError(f"Pose model not found: {model_file}")

    cache_path = cache_path_for(video, model_file, cache_dir)
    if cache_path.is_file() and not force:
        sequence = load_reference(cache_path)
        if progress is not None:
            progress(sequence.frame_count, sequence.frame_count)
        return sequence

    # Imported lazily so scoring and pose utilities remain testable without YOLO.
    from ultralytics import YOLO

    model = YOLO(str(model_file))
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open reference video: {video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1e-3:
        fps = 30.0
    expected_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    poses: list[Pose] = []
    timestamps: list[float] = []
    smoother = PoseSmoother(alpha=0.68, gap_decay=0.48)
    previous_center: Optional[np.ndarray] = None
    index = 0
    try:
        while capture.isOpened():
            if cancel_event is not None and cancel_event.is_set():
                raise ReferenceAnalysisCancelled("Reference analysis cancelled")
            ok, frame = capture.read()
            if not ok:
                break
            result = model.predict(
                source=frame,
                conf=0.25,
                imgsz=640,
                max_det=8,
                verbose=False,
            )[0]
            selection = select_target_person(
                result,
                frame.shape,
                previous_center=previous_center,
                conf_threshold=0.25,
            )
            pose = pose_from_result(result, selection, conf_threshold=0.25)
            pose = smoother.update(pose)
            if pose is None:
                from .pose_utils import empty_pose

                pose = empty_pose()
            if np.isfinite(pose.center).all():
                previous_center = pose.center
            poses.append(pose)
            timestamps.append(index / fps)
            index += 1
            if progress is not None and (
                index == 1 or index % 10 == 0 or index == expected_count
            ):
                progress(index, expected_count)
    finally:
        capture.release()

    if not poses:
        raise RuntimeError("No readable frames were found in the reference video")
    sequence = ReferenceSequence(
        poses=poses,
        timestamps=np.asarray(timestamps, dtype=np.float64),
        fps=fps,
        frame_count=len(poses),
        width=width,
        height=height,
        source_path=str(video),
    )
    save_reference(sequence, cache_path)
    return sequence
