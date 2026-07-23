# Bonus Level Requirement and Acceptance Plan

This checklist converts the assignment PDF into implementation and demo
acceptance criteria.

## Task 1 - TikTok Dance Analysis

- [x] Use a pretrained body-movement detector; no dataset training is required.
- [x] Support full-body and partial-body detections.
- [x] Visualize reliable keypoints and skeleton edges.
- [x] Choose a primary dancer when multiple people appear.
- [x] Filter low-confidence joints instead of mapping them to `(0, 0)`.
- [x] Smooth short detection gaps and jitter.
- [x] Document challenges, strategies, and representative examples.
- [ ] Team action: add 3 course-dataset screenshots to the final presentation.

Acceptance demo: run detection on one clean full-body clip, one cropped or
occluded clip, and one clip containing multiple people. Confirm that the main
dancer remains selected and missing joints do not create lines to the corner.

## Task 2 - Just Dance

- [x] Apply body detection to reference video and live webcam.
- [x] Display movements in both GUI panels.
- [x] Allow reference source video to be shown or hidden.
- [x] Normalize poses spatially for camera position and scale.
- [x] Align poses temporally and account for ordinary webcam/reaction lag.
- [x] Use multiple scoring metrics rather than raw point distance alone.
- [x] Display Perfect, Super, Good, and Miss feedback.
- [x] Display numeric score, session average, coverage, lag, and FPS.
- [x] Provide mirror imitation mode.
- [x] Cache the reference analysis for repeatable demos.
- [x] Keep GUI updates on the Tkinter main thread.
- [x] Add deterministic tests for the scoring assumptions.
- [ ] Optional extension: beat analysis with librosa/madmom.
- [ ] Optional extension: a Fruit-Ninja-style bonus activity.

Acceptance demo:

1. Analyze a 10-30 second reference clip.
2. Start the game and confirm both skeletons are visible.
3. Move closer/farther and left/right; the same pose should remain comparable.
4. Imitate a pose with a small delay; the lag display should respond.
5. Hide the source video; the reference skeleton should remain visible.
6. Step partly out of frame; coverage should fall without a false corner joint.
7. Stop and restart; the app should release and reopen the camera cleanly.

## Presentation explanation

Use this sequence for the technical story:

1. **Detection:** YOLOv8-pose returns 17 COCO keypoints and confidence values.
2. **Dancer selection:** rank people using coverage, size, confidence, centrality,
   and temporal continuity.
3. **Cleaning:** confidence mask plus short confidence-decaying EMA smoothing.
4. **Spatial alignment:** hip-center translation and torso/body-scale division.
5. **Temporal alignment:** local timestamp search in a `±450 ms` window.
6. **Score:** 50% angle, 35% shape/limb direction, 15% motion, adjusted by
   shared-body coverage.
7. **Feedback:** threshold the final score and show component-level diagnostics.

The core design favors explainability and reliable local CPU execution over a
large learned similarity network, which is appropriate because the assignment
does not require training and does not provide a GPU.
