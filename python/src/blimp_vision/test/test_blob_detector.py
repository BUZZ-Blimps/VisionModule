"""
Tests for the HSV balloon detector (blob_detector.py) on synthetic frames.

Runs without ROS: if blimp_vision_msgs is not installed, a minimal stand-in
for the Detection message is used.

    python3 -m pytest python/src/blimp_vision/test/test_blob_detector.py
"""

import sys
import types

import cv2
import numpy as np
import pytest

try:
    from blimp_vision_msgs.msg import Detection  # noqa: F401
except ImportError:
    class Detection:
        def __init__(self):
            self.class_id = 0
            self.obj_class = ''
            self.bbox = np.zeros(4, dtype=np.float32)
            self.depth = 0.0
            self.confidence = 0.0
            self.track_id = 0

    msgs = types.ModuleType('blimp_vision_msgs')
    msgs.msg = types.ModuleType('blimp_vision_msgs.msg')
    msgs.msg.Detection = Detection
    sys.modules['blimp_vision_msgs'] = msgs
    sys.modules['blimp_vision_msgs.msg'] = msgs.msg

from blimp_vision.blob_detector import BlobDetectorClass  # noqa: E402

WIDTH, HEIGHT = 640, 480
PURPLE_HSV = (120, 120, 150)  # Inside the default purple range
GREEN_HSV = (55, 120, 150)    # Inside the default green range


def hsv_to_bgr(hsv):
    pixel = np.uint8([[hsv]])
    return tuple(int(c) for c in cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0])


def make_frame(center=None, radius=30, hsv=PURPLE_HSV):
    frame = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)
    if center is not None:
        cv2.circle(frame, center, radius, hsv_to_bgr(hsv), -1)
    return frame


def run(detector, frames):
    return [detector.contour_find_ball(frame) for frame in frames]


def test_detects_still_balloon():
    detector = BlobDetectorClass()
    det = run(detector, [make_frame((320, 300))] * 5)[-1]
    assert det is not None
    assert abs(det.bbox[0] - 320) < 5
    assert abs(det.bbox[1] - 300) < 5


def test_detects_green_balloon():
    detector = BlobDetectorClass()
    det = run(detector, [make_frame((320, 300), hsv=GREEN_HSV)] * 5)[-1]
    assert det is not None


def test_keeps_detecting_after_lock_on():
    # A balloon held still for 40 frames triggers lock-on (after ~20 frames).
    # The balloon must still be measured, not just predicted, after locking.
    # The blimp's catch triggers use the box area, so the size must stay right too.
    detector = BlobDetectorClass()
    det = run(detector, [make_frame((320, 300), radius=30)] * 40)[-1]
    assert detector.lock_on
    assert detector.x > -1, 'Balloon lost after lock-on'
    assert abs(det.bbox[2] - 60) < 10, f'Balloon diameter {det.bbox[2]:.0f}px after lock-on, expected ~60px'


def test_follows_balloon_that_moves_after_lock_on():
    detector = BlobDetectorClass()
    run(detector, [make_frame((320, 300))] * 30)
    assert detector.lock_on
    results = run(detector, [make_frame((320 + 8 * i, 300)) for i in range(1, 16)])
    assert detector.x > -1, 'Balloon lost once it moved after lock-on'
    assert abs(results[-1].bbox[0] - (320 + 8 * 15)) < 15


def test_optical_flow_velocity_sign():
    # Balloon moving right/down in the image should give positive velocity.
    # Textured balloon so optical flow has corners to follow.
    detector = BlobDetectorClass()
    for i in range(10):
        frame = make_frame((200 + 5 * i, 200 + 5 * i), radius=40)
        cv2.circle(frame, (200 + 5 * i - 10, 200 + 5 * i - 10), 6, hsv_to_bgr((125, 200, 110)), -1)
        cv2.circle(frame, (200 + 5 * i + 12, 200 + 5 * i + 8), 5, hsv_to_bgr((115, 90, 175)), -1)
        detector.contour_find_ball(frame)
    assert detector.v_x_s > 0, f'x velocity {detector.v_x_s:.2f} should be positive'
    assert detector.v_y_s > 0, f'y velocity {detector.v_y_s:.2f} should be positive'


def test_drops_target_soon_after_balloon_disappears():
    # The blimp has its own short memory, so the detector should not keep
    # reporting a balloon for seconds after it leaves the frame.
    detector = BlobDetectorClass()
    run(detector, [make_frame((320, 300))] * 10)
    results = run(detector, [make_frame(None)] * 10)
    assert results[-1] is None, 'Phantom detection 10 frames after balloon left'


def test_close_balloon_does_not_change_tuned_values():
    detector = BlobDetectorClass()
    tuned_purple_uv = detector.purple_uv
    tuned_green_uv = detector.green_uv
    run(detector, [make_frame((320, 300), radius=60)] * 5)
    assert detector.purple_uv == tuned_purple_uv
    assert detector.green_uv == tuned_green_uv


def test_tracks_moving_balloon_in_both_directions():
    for step in (6, -6):
        detector = BlobDetectorClass()
        start = 200 if step > 0 else 440
        frames = [make_frame((start + step * i, 300)) for i in range(30)]
        results = run(detector, frames)
        assert all(r is not None for r in results[3:]), f'Lost moving balloon (step {step})'
        final_x = start + step * 29
        assert abs(results[-1].bbox[0] - final_x) < 15, f'Estimate lags/leads (step {step})'


def test_ignores_non_balloon_shapes():
    frame = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)
    cv2.rectangle(frame, (100, 300), (500, 320), hsv_to_bgr(PURPLE_HSV), -1)  # Long thin bar
    detector = BlobDetectorClass()
    assert run(detector, [frame] * 5)[-1] is None


@pytest.mark.parametrize('ratio,expected', [(0.0, True), (0.5, False)])
def test_ignore_top_ratio(ratio, expected):
    detector = BlobDetectorClass()
    detector.ignore_top_ratio = ratio
    det = run(detector, [make_frame((320, 100))] * 5)[-1]
    assert (det is not None) == expected


def test_reports_confidence():
    detector = BlobDetectorClass()
    det = run(detector, [make_frame((320, 300))] * 5)[-1]
    assert 0.0 < det.confidence <= 1.0


def test_reports_true_size_before_lock_on():
    detector = BlobDetectorClass()
    det = run(detector, [make_frame((320, 300), radius=30)] * 5)[-1]
    assert abs(det.bbox[2] - 60) < 6, f'Balloon diameter {det.bbox[2]:.0f}px, expected ~60px'


def test_detects_balloon_with_glare_spot():
    frame = make_frame((320, 300), radius=40)
    cv2.circle(frame, (310, 290), 7, (255, 255, 255), -1)  # Light reflection
    detector = BlobDetectorClass()
    det = run(detector, [frame] * 5)[-1]
    assert det is not None
    assert abs(det.bbox[2] - 80) < 8


def test_reacquires_balloon_in_new_place_after_losing_it():
    # A lock-on must not keep searching only the old spot once the balloon is gone.
    detector = BlobDetectorClass()
    run(detector, [make_frame((200, 300))] * 30)
    assert detector.lock_on
    run(detector, [make_frame(None)] * 10)
    det = run(detector, [make_frame((480, 300))] * 5)[-1]
    assert det is not None
    assert abs(det.bbox[0] - 480) < 5
