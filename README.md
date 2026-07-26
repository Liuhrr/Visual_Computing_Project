# NUS Visual Computing Project - Just Dance Pose Match

A local, real-time dance game built for the **Bonus Level** of the NUS Visual
Computing project. YOLOv8-pose detects 17 COCO body keypoints in a reference
video and a webcam stream. The app selects the main dancer, aligns the player's
pose in space and time, and displays a live score with
**Perfect / Super / Good / Miss** feedback.

## What is implemented

- Main-dancer selection for videos containing multiple people
- Confidence filtering and short-gap keypoint smoothing
- Full-body and partial-body scoring without treating missing joints as `(0, 0)`
- Cached reference-video analysis: each reference frame is inferred only once
- One webcam inference per submitted frame in a background worker
- Translation and body-scale normalization
- Optional mirrored choreography matching with left/right joint remapping
- Local temporal alignment within `±450 ms` to account for reaction/camera lag
- Composite angle, normalized-shape, limb-direction, and motion scoring
- Reference video / skeleton-only toggle
- Live score, feedback, average score, lag, coverage, and application FPS
- Restrained responsive Tkinter interface with eight explicit UI states
- Session summary with Perfect, Super, Good, and Miss pose counts
- Thread-safe Tkinter updates and clean camera/model shutdown

## Frontend design

The interface follows a **Quiet Studio** direction: warm off-white canvas,
white content surfaces, charcoal typography, thin borders, and one muted sage
accent. Reference and player video remain the dominant elements. Scores and
diagnostics live in a separate horizontal performance panel and never cover the
dancer.

The UI is designed around these explicit states:

| State | User-facing behavior |
|---|---|
| Empty | Prompts the user to choose a reference video |
| Video Loaded | Shows the first frame, filename, and duration |
| Analyzing | Shows frame progress and a thin progress bar |
| Ready | Enables Start Dance and reports reference pose coverage |
| Running | Streams both views and updates each new inferred pose once |
| Pose Lost | Pauses scoring and asks the player to step back |
| Camera Error | Shows an in-panel error and a Retry Camera action |
| Finished | Shows average score, feedback counts, and Dance Again |

Buttons, status text, and colors change together, so no important state depends
on color alone. The two video panels resize evenly while preserving the source
aspect ratio.

## Quick start with the provided Conda environment

```bash
cd /Users/quchengzou/Desktop/NUS_Project/Visual_Computing_Project_wall_work_2026-07-25-225019
conda activate nus
python danceapp.py
```

The current `nus` environment already contains all required packages. For a new
environment:

```bash
python -m pip install -r requirements.txt
```

The bundled model must remain at:

```text
models/yolov8n-pose.pt
```

### Game workflow

1. Click **Open Video** and choose a short reference dance video.
2. Click **Analyze Reference**. The first run performs pose inference; later
   runs load an automatically invalidated cache.
3. Stand far enough from the camera for shoulders, hips, and limbs to be visible.
4. Leave **Mirror choreography** enabled when copying the dancer as if looking
   into a mirror. Disable it when matching anatomical left/right directly.
5. Click **Start Dance**. The reference video and webcam start together.
6. Use **Show video** to switch the left panel between the source frame and a
   skeleton-only view. Click **Stop** to end the session.

## How the score works

The scorer does not compare raw pixel coordinates directly.

### 1. Reliable joints only

A joint participates only when it is detected confidently in both poses.
Invisible or occluded joints are stored as `NaN` and ignored. The coverage value
reports how many of the 12 body joints (shoulders through ankles) are shared.

### 2. Spatial alignment

Each pose is translated to its hip center (shoulder center is the fallback) and
scaled by torso/shoulder/hip size. This makes the score largely invariant to the
player's location, camera resolution, and distance from the camera.

### 3. Composite similarity

Available components are reweighted if a partial-body pose cannot support one:

```text
raw score = 0.50 × joint-angle score
          + 0.35 × normalized shape / limb-direction score
          + 0.15 × motion score

final score = raw score × (0.72 + 0.28 × body coverage)
```

- **Angle:** confidence-weighted similarity of elbows, shoulders, hips, knees
- **Shape:** normalized keypoint distance plus skeleton-bone direction
- **Motion:** direction and speed consistency between consecutive poses

The feedback thresholds are:

| Final score | Feedback |
|---:|:---|
| `≥ 90%` | Perfect! |
| `75% - 89.9%` | Super! |
| `60% - 74.9%` | Good |
| `< 60%` | Miss |

### 4. Temporal alignment

At time `t`, the scorer searches only the reference poses in
`[t - 0.45 s, t + 0.45 s]`. Pose similarity chooses the best local match, while
a small distance penalty prevents unnecessary jumps. The estimated lag is
smoothed before display. This handles ordinary human reaction and camera
latency without matching an unrelated move from a distant part of the video.

## Mapping to the assignment

| Assignment item | Implementation |
|---|---|
| Task 1: full/partial body detection | YOLOv8-pose, confidence masks, partial-body scoring |
| Multiple people challenge | coverage/size/center/confidence/continuity target ranking |
| Detect and visualize movement | cached reference poses and live player skeleton |
| Apply pipeline to video + webcam | synchronized dual-panel game |
| Design a scoring system | spatial normalization, local temporal search, composite metrics |
| Account for webcam lag | rolling `±450 ms` alignment and displayed lag |
| Display user feedback | Perfect, Super, Good, Miss, score and average |
| Explain metrics in presentation | formula and rationale documented above |

The TikTok dataset is used for testing and improving detection, not for model
training, matching the assignment note. The app can use any suitable dance
video; TikTok videos are not mandatory for Task 2.

## Task 1 findings to present

The most important practical failure modes and the implemented responses are:

- **Multiple people:** choosing detection index zero is unstable. Rank candidates
  by visible-joint coverage, body size, detector confidence, centrality, and
  continuity with the previous dancer.
- **Partial body / occlusion:** low-confidence points must not be drawn at the
  image origin or converted to zero angles. Preserve missing values and compare
  only reliable shared joints.
- **Jitter:** use confidence-aware exponential smoothing and retain a missing
  joint for only a short, decaying gap.
- **Different camera framing:** compare hip-centered, body-scale-normalized poses
  rather than pixels.
- **Left/right imitation:** provide an explicit mirror mode that also swaps COCO
  left/right labels.
- **Lag and frame-rate differences:** use timestamp-based local alignment rather
  than frame-number equality.

For the presentation, show a few representative TikTok frames: one clear
full-body example, one partial/occluded example, and one multi-person example.
The assignment explicitly says a full-dataset evaluation is not required.

## Project structure

```text
Visual_Computing_Project/
├── danceapp.py                 # Tkinter game and live webcam pipeline
├── models/yolov8n-pose.pt
├── utils/
│   ├── pose_utils.py           # selection, extraction, smoothing, normalization
│   ├── reference.py            # reference analysis and .npz cache
│   └── scoring.py              # spatial/temporal scoring and feedback
├── wall_config.py              # tunables for the bonus activity
├── wall_game.py                # wall state machine, rendering, FREEZE mini-game
├── extract_wall_poses.py       # offline iconic-pose extraction from reference video
├── scripts/demo_wall.py        # headless demo saving key frames
├── tests/                      # deterministic synthetic-pose tests
├── data/TikTokDataset/         # supplied metadata; local videos are ignored
├── PROJECT_PLAN.md             # requirement checklist and demo acceptance plan
└── requirements.txt
```

Generated reference pose files are written to `cache/reference/` and ignored by
Git. A cache is rebuilt automatically if the video size/modification time,
model filename, or cache format changes.

## Verification

Run the deterministic test suite:

```bash
conda activate nus
python -m pytest -q
```

Run a model smoke test on a real image or video frame:

```bash
python scripts/smoke_test_model.py --source /path/to/video.mp4
```

The smoke test writes an annotated image under `outputs/smoke_test/`.

## Bonus activity: Hole in the Wall + FREEZE

This build adds an optional bonus activity on top of the core Just Dance scorer.
All bonus code lives in new files; the original scoring pipeline is unchanged.

### What was added

- **Wall poses (`extract_wall_poses.py`)**: automatically picks 8–15 iconic poses
  from the reference video by finding low-motion, high-confidence intervals.
- **Wall game (`wall_game.py`)**: procedural grey wall with a human-shaped
  transparent cutout, guide skeleton, approach animation, judgement line, and
  PASS/FAIL particle effects.
- **Pose matching**: uses the existing `compute_pose_score` and
  `normalize_pose` functions (no new model).
- **FREEZE mini-game**: music-stops event where the player must hold still;
  movement is measured by accumulated keypoint displacement.
- **Integration (`danceapp.py`)**: bonus score and combo are displayed live and
  summarized in the finished state as Dance / Bonus / Total / Grade.

### Running the bonus activity

1. Generate wall poses and schedule from a reference video (done once per video):

   ```bash
   conda activate nus
   python extract_wall_poses.py data/dance_example_1.mp4
   ```

   The schedule is created automatically the first time you click
   **Analyze Reference** or **Start Dance**.

2. Run the main app with walls enabled (default):

   ```bash
   python danceapp.py
   ```

3. Headless demo (no camera, saves key frames to `outputs/wall_demo/`):

   ```bash
   python danceapp.py --demo --demo-save-frames --video data/dance_example_1.mp4
   ```

   A standalone demo script is also available:

   ```bash
   python scripts/demo_wall.py
   ```

### Command-line flags

| Flag | Effect |
|---|---|
| `--demo` | Synthetic poses, no camera required |
| `--demo-save-frames` | Save SPAWN/APPROACH/JUDGE/PASS/FAIL frames |
| `--no-wall` | Disable the wall bonus |
| `--no-freeze` | Disable the FREEZE mini-game |
| `--video PATH` | Pre-load a reference video on startup |

### Tunables (see `wall_config.py`)

- `WALL_APPROACH_SECONDS`: how long the wall takes to reach the judge line.
- `WALL_SIMILARITY_PERFECT` / `WALL_SIMILARITY_GOOD`: judgement thresholds.
- `WALL_TIME_WINDOWS`: intervals where walls are allowed to spawn.
- `WALL_INTERVAL_SECONDS`: fixed-interval fallback if `librosa` is unavailable.
- `FREEZE_EVENTS`: list of `{start, duration}` silence events.
- `FREEZE_DISPLACEMENT_THRESHOLD`: movement tolerance in pixels.

### Web-based Bonus Level frontend

A browser-based frontend is available in `web_bonus/`. It connects to the **same**
existing pose detection, scoring, wall, and FREEZE logic via a small Flask
backend (`web_app.py`).

```bash
conda activate nus
python web_app.py
```

Then open:

```text
http://127.0.0.1:8080
```

The server auto-loads `data/dance_example_1.mp4` on startup (if present) so you
can click **Start Game** immediately. You can also upload your own reference
video; analysis runs in the background and the UI enables the start button as
soon as it is ready.

What the web frontend provides:

- Landing page with rules
- Reference video upload / auto-load
- Real-time camera stream with processed overlays
- Live HUD: Dance score, Bonus score, Combo, Feedback, visible joints, FPS
- Wall badge and FREEZE overlay
- Stop and final-result screen with grade

### Demo / test output

- `outputs/wall_demo/` contains representative frames for each wall state and
  FREEZE outcome.
- `cache/wall_poses.json` and `cache/wall_schedule.json` cache the extracted
  poses and spawn times.

## Known limitations and sensible scope

- YOLOv8 provides 2D pose only; severe self-occlusion and depth ambiguity remain.
- The app intentionally does not train a network, because the assignment does
  not require training and no course GPU is provided.
- Playback uses OpenCV and therefore does not output the video's audio. Beat
  analysis and a bonus activity are optional assignment extensions, not core
  scoring requirements.
- A two-person duet would need separate dancer tracks and assignment logic; this
  implementation deliberately selects one primary dancer.

## Open-source basis

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) provides the
  pose detector and pretrained weights.
- [DanceSync](https://github.com/takatoshilee/DanceSync) informed the general idea
  of pose-based dance comparison; this repository's scoring and GUI are a new,
  compact implementation tailored to the assignment.

Review the relevant upstream licenses before redistributing the application
outside coursework.
