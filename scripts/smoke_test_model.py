"""Run one real YOLO pose inference and save the annotated result."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault(
    "MPLCONFIGDIR", str(PROJECT_ROOT / "cache" / "matplotlib")
)

from utils.pose_utils import (  # noqa: E402
    draw_pose,
    pose_from_result,
    select_target_person,
    valid_keypoints,
)


def candidate_frames(source: Path):
    image = cv2.imread(str(source))
    if image is not None:
        yield 0, image
        return
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to read image or video: {source}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = sorted(
        {
            0,
            frame_count // 10,
            frame_count // 4,
            frame_count // 2,
            3 * frame_count // 4,
        }
    )
    try:
        for frame_index in sample_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if ok:
                yield frame_index, frame
    finally:
        capture.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "smoke_test" / "pose.jpg",
    )
    args = parser.parse_args()

    from ultralytics import YOLO

    source = args.source.expanduser().resolve()
    model = YOLO(str(PROJECT_ROOT / "models" / "yolov8n-pose.pt"))
    frame = None
    frame_index = 0
    result = None
    selection = None
    pose = None
    for frame_index, candidate in candidate_frames(source):
        candidate_result = model.predict(
            source=candidate, conf=0.25, imgsz=640, max_det=8, verbose=False
        )[0]
        candidate_selection = select_target_person(
            candidate_result, candidate.shape
        )
        candidate_pose = pose_from_result(
            candidate_result, candidate_selection
        )
        if candidate_pose is not None:
            frame = candidate
            result = candidate_result
            selection = candidate_selection
            pose = candidate_pose
            break
    if pose is None:
        print("No reliable person detected in sampled frames.")
        return 2
    visible = int(valid_keypoints(pose).sum())
    annotated = draw_pose(frame, pose, label="SELECTED DANCER")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), annotated):
        raise RuntimeError(f"Unable to write: {args.output}")
    people = len(result.keypoints.xy) if result.keypoints is not None else 0
    print(
        f"OK: detected {people} person(s), selected index "
        f"{selection.index if selection else 0}, {visible}/17 visible keypoints, "
        f"source frame {frame_index}"
    )
    print(f"Annotated frame: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
