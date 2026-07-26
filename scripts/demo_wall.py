"""Headless demo of the wall game and FREEZE mini-game.

Generates synthetic player poses (matching and deliberately wrong) and saves
key frames for visual inspection without requiring a camera or display.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.pose_utils import Pose, valid_keypoints  # noqa: E402
from extract_wall_poses import load_wall_poses  # noqa: E402
from wall_config import DEMO_OUTPUT_DIR  # noqa: E402
from wall_game import FreezeGame, Wall, WallGame, WallState, build_wall_schedule  # noqa: E402


def _make_matching_pose(target: Pose, noise: float = 2.0) -> Pose:
    perturbation = np.random.normal(0.0, noise, target.xy.shape).astype(np.float32)
    return Pose(xy=target.xy + perturbation, confidence=target.confidence.copy())


def _make_wrong_pose(target: Pose) -> Pose:
    xy = target.xy.copy()
    valid = valid_keypoints(target)
    # Raise arms straight up to create a strong mismatch
    for idx in (7, 8, 9, 10):
        if valid[idx]:
            xy[idx, 1] -= 140.0
    return Pose(xy=xy, confidence=target.confidence.copy())


def _demo_wall_cycle(
    wall: Wall,
    should_pass: bool,
    output_dir: Path,
    prefix: str,
) -> None:
    """Run one wall from spawn through judgement and save key frames."""
    saved: set[str] = set()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)

    # Advance time in 33ms steps
    elapsed = wall.spawn_time
    while True:
        state = wall.update(elapsed)

        user_pose: Pose | None = None
        if state in (WallState.APPROACH, WallState.JUDGE):
            user_pose = _make_matching_pose(wall.target_pose) if should_pass else _make_wrong_pose(wall.target_pose)

        if state == WallState.JUDGE and not wall.judged:
            event = wall.judge(user_pose)
            print(f"  judgement: {event.kind}  score={event.score:+}")
            if event.kind in ("perfect", "good"):
                wall.mark_pass(elapsed)
            else:
                wall.mark_fail(elapsed)
            # Render immediately after judgement so the saved frame shows the result
            rendered = wall.render(frame.copy(), elapsed, user_pose)
            label = "judge"
            if label not in saved:
                saved.add(label)
                path = output_dir / f"{prefix}_{label}_{elapsed:.2f}.png"
                cv2.imwrite(str(path), rendered)
                print(f"Saved {path}")
        else:
            rendered = wall.render(frame.copy(), elapsed, user_pose)
            label = state.value
            if label not in saved:
                saved.add(label)
                path = output_dir / f"{prefix}_{label}_{elapsed:.2f}.png"
                cv2.imwrite(str(path), rendered)
                print(f"Saved {path}")

        if state == WallState.IDLE:
            break

        elapsed += 0.033


def main() -> None:
    poses = load_wall_poses(PROJECT_ROOT / "cache" / "wall_poses.json")
    if not poses:
        raise RuntimeError("No wall poses found. Run extract_wall_poses.py first.")

    schedule = build_wall_schedule(
        PROJECT_ROOT / "data" / "dance_example_1.mp4",
        poses,
    )

    output_dir = DEMO_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Demo 1: deliberately fail the first wall
    print("--- FAIL demo ---")
    wall_fail = Wall(_pose_from_dict(poses[0]), spawn_time=0.0)
    _demo_wall_cycle(wall_fail, should_pass=False, output_dir=output_dir, prefix="fail")

    # Demo 2: pass the second wall
    print("--- PASS demo ---")
    wall_pass = Wall(_pose_from_dict(poses[1]), spawn_time=0.0)
    _demo_wall_cycle(wall_pass, should_pass=True, output_dir=output_dir, prefix="pass")

    # Demo 3: FREEZE event
    print("--- FREEZE demo ---")
    freeze = FreezeGame(events=[{"start": 0.0, "duration": 0.5}], enabled=True)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)
    target = _pose_from_dict(poses[0])
    saved_freeze = False
    for i, t in enumerate(np.arange(0.0, 0.6, 0.033)):
        # First half: hold still; second half: move a lot
        if t < 0.25:
            pose = _make_matching_pose(target, noise=0.5)
        else:
            pose = _make_wrong_pose(target)
        rendered, event = freeze.update(t, pose, frame.copy())
        if not saved_freeze and freeze.active_index is not None:
            saved_freeze = True
            path = output_dir / f"freeze_active_{t:.2f}.png"
            cv2.imwrite(str(path), rendered)
            print(f"Saved {path}")
        if event is not None:
            path = output_dir / f"freeze_result_{event.kind}_{t:.2f}.png"
            cv2.imwrite(str(path), rendered)
            print(f"Saved {path}  score={event.score:+}")
            break

    print(f"\nDemo frames written to {output_dir}")


def _pose_from_dict(data: dict) -> Pose:
    return Pose(
        xy=np.asarray(data["xy"], dtype=np.float32),
        confidence=np.asarray(data["confidence"], dtype=np.float32),
    )


if __name__ == "__main__":
    main()
