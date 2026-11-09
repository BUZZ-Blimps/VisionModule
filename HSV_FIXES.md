# HSV Detection Fixes (branch `hsv-fixes`)

The HSV (color-based) detector in `python/src/blimp_vision` was reported as not working on the blimp. This branch fixes the problems found in the code. It has **not yet been tested on a blimp**.

## How the HSV pipeline works

```
Stereo camera (1280x480) → camera_node.py → left half (640x480)
    Ball search:  blob_detector.py (purple/green balloons)
    Goal search:  contour_goal_detection.py (orange/yellow goals), YOLO goal model as backup
→ /<Name>/targets (Float64MultiArray, 9 values) → CatchingBlimp (BlimpCore/ros2_ws)
```

`/targets` is `[x, y, depth, track_id, type (0 ball / 1 goal), theta_x, theta_y, box_w, box_h]`, or all `-1` when nothing is detected.

In ball search, the blimp steers with `x`, `y` and the angles. It opens the gate and starts catching based on the **box area** (`BALL_GATE_OPEN_TRIGGER = 2000`, `BALL_CATCH_TRIGGER = 12500` px² in `CatchingBlimp.hpp`), so the reported balloon size directly controls when it catches. Depth isn't used for catching.

## What was wrong

These were confirmed with tests on synthetic frames (`test/test_blob_detector.py`). The original code fails 10 of the 15 tests: 8 bugs, plus 2 new features.

| # | Problem | Effect on the blimp | Status |
|---|---|---|---|
| 1 | **Lock-on shrinks the search area to a 4 px circle.** After ~20 steady frames (~1.5 s), the detector only searched a circle of radius `lock_radius * 2` = 4 px around the last position. | The balloon was reported as ~21 px wide instead of 60 px, so its box area never reached the gate (2000 px²) or catch (12500 px²) triggers. Once the balloon moved more than a few pixels, it was lost entirely. | Fixed. The radius is now `lock_radius` × the balloon's radius. |
| 2 | **Lock-on never released while the balloon was gone.** Kalman predictions kept counting as "same place", so lock-on stayed on. | A balloon appearing somewhere else couldn't be detected until the filter reset. | Fixed. Only real detections count toward lock-on, and tracking resets when the balloon is lost. |
| 3 | **Phantom balloon for ~6.7 s.** After the balloon left the frame, the detector kept reporting Kalman predictions for up to 100 frames. | The blimp chased a balloon that wasn't there. The blimp already has its own short memory (0.5 s detect timeout, 2 s memory). | Fixed. Detections stop after `max_coast_frames` (5 frames, ~0.3 s). |
| 4 | **Optical flow x-velocity had the wrong sign** (`old - new` instead of `new - old`). y was correct. | The Kalman filter's motion estimate pulled sideways predictions the wrong way. | Fixed. |
| 5 | **Tuned brightness limit overwritten permanently.** Once a balloon looked big (radius > 40 px), `purple_uv` and `green_uv` were set to 250 and never restored. | After the first close balloon, the tuned values were gone for the rest of the run, so more background could match. | Fixed. The higher limit is applied per frame without changing the tuned values. |
| 6 | **Mask cleanup inflated balloon size.** Erode 3–4 px then dilate 10–13 px made balloons 10–25% wider. | The gate and catch triggers fired earlier than intended. | Fixed. Uses opening + closing, which removes specks and fills glare holes without changing size. |
| 7 | **Kalman reset only on jumps in both x and y** of more than 10 px. | A jump to a different balloon along one axis was blended in instead of resetting. | Fixed. Resets on a straight-line jump larger than `max(20 px, balloon radius)`. |
| 8 | **Lock-on widened HSV bounds past valid ranges** (e.g. hue above 179). | Mostly harmless, but undefined. | Fixed. Bounds are clipped. |
| 9 | **Missing YOLO goal model crashes the node at startup.** All launch files load `models/goal/goal_v11n_rknn_model`, which isn't in the repo; only `goal_v11s` exists. Both models load at startup even though ball search only uses HSV. | Likely the main reason the node "doesn't run": it crashes before detecting anything. | Fixed in code: a model that fails to load is logged and skipped, and goal search falls back to HSV. **Still to decide:** add `goal_v11n` or point the launch files at `goal_v11s` (see next steps). Not tested on hardware. |
| 10 | **Debug video stream crashes the node if it can't start**, for example when the Rockchip encoder plugin `mpph264enc` is missing. The destination IP was also hard-coded. | Node crashes at startup on a machine without the plugin. | Fixed: a stream failure is logged and detection continues. Added `stream_host` / `stream_port` parameters with the old values as defaults. Not tested on hardware. |
| 11 | **Camera open failure left a half-built node.** The node logged the error but kept going, then crashed later with an unrelated error. | Confusing error messages. | Fixed: it now fails immediately with a clear error. |

Smaller cleanups:
- `blob_detector.py`: removed a leftover `cv.VideoCapture('output.avi')` in the constructor; `isValidArea` no longer overwrites the stored position as a side effect; masks are combined with `bitwise_or` instead of `+`.
- `camera_node.py`: state changes are compared with `!=` instead of `is not`; removed unused imports.

## What was added

- **Confidence score:** HSV detections now report a confidence (how fully the balloon fills its enclosing circle, 0–1) instead of `-1`. It's in the `Detection` message but not yet in `/targets`. It's a starting point for choosing between YOLO and HSV later.
- **`ignore_top_ratio`:** ignore detections in the top part of the frame, for ceiling lights. Off by default (`0.0`), so behavior is unchanged unless set. `contour_ball_detection.py` had a hard-coded version of this, but that file isn't used.
- **New tunable values** on `BlobDetectorClass`: `close_radius`, `close_uv`, `lock_frames`, `max_coast_frames`.

All existing attribute names were kept, so the YAML tuning loader on the `yaml-loader` branch (which sets `blob_detector.purple_uv`, `ignore_top_ratio`, etc.) still works with these changes.

## Testing

Synthetic-frame tests, which need no ROS or camera, only `opencv-python`, `numpy` and `pytest`:

```bash
cd python/src/blimp_vision
PYTHONPATH=. python3 -m pytest test/test_blob_detector.py
```

They cover detection of still, moving and green balloons; correct size before and after lock-on; following a balloon after lock-on; re-finding a balloon in a new place; dropping a lost balloon quickly; the velocity sign; tuned values staying unchanged; glare spots; rejecting non-round shapes; `ignore_top_ratio`; and confidence.

**Not tested:**
- The `camera_node.py` changes (startup handling, streaming), which need ROS 2, the stereo camera and the Rockchip GStreamer plugin.
- Real footage. The default HSV ranges in `blob_detector.py` and `contour_goal_detection.py` haven't been checked against current balloons, goals or lighting.

## Next steps

1. **Run on a blimp.** Build and launch as usual, e.g. `ros2 launch blimp_vision 1_BurnCream.launch.py`.
   - Check the log for model or stream errors.
   - Run `ros2 topic echo /BurnCream/targets` while holding a balloon in front of the camera. Values should be non-negative, and `box_w`/`box_h` should grow as the balloon gets closer.
2. **Decide the goal model with Nathan.** Either add `goal_v11n_rknn_model` to `models/goal/` or change the launch files to `goal_v11s_rknn_model` (bigger and slower). Until then, goal search uses HSV only.
3. **Tune the HSV ranges** for current balloons and lighting using `BuzzBlimpsHSVTuner`, and record a few videos (`save_frames:=true`) to re-test without flying.
4. **Merge or rebase the `yaml-loader` branch** so tuned values load from a YAML file instead of being hard-coded. It should be compatible with this branch.
5. **Check the catch triggers.** Balloon sizes are now accurate rather than inflated, so `BALL_GATE_OPEN_TRIGGER` / `BALL_CATCH_TRIGGER` in BlimpCore may need re-tuning in flight.
6. **Clean up later:**
   - `contour_ball_detection.py` is unused and duplicates the HSV ball code; remove it or merge its top-of-frame filter (now `ignore_top_ratio`).
   - Add confidence to `/targets` or a new topic, for YOLO/HSV arbitration.
