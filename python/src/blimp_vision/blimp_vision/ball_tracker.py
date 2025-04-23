#!/usr/bin/env python3
"""
Module for BallTracker, which selects the optimal target from detection results.
"""

import numpy as np
from blimp_vision_msgs.msg import Detection


class BallTracker:
    def __init__(self, width, height):
        """
        Initialize the BallTracker.
        
        :param width: Frame width.
        :param height: Frame height.
        """
        self.current_tracked_id = None
        self.frame_center = (int(width // 2), int(height // 2))
        self.lost_track_threshold = 5  # Frames to wait before declaring track lost.
        self.frames_without_target = 0
        self.goal_cutoff_index = 3.0

    def calculate_center_distance(self, detection):
        """
        Calculate Euclidean distance from detection center to frame center.
        
        :param detection: A list or array [x, y, w, h].
        :return: Distance as a float.
        """
        return ((detection[0] - self.frame_center[0]) ** 2 +
                (detection[1] - self.frame_center[1]) ** 2) ** 0.5

    def calculate_area(self, detection):
        """
        Calculate the area of the detection bounding box.
        
        :param detection: A list or array [x, y, w, h].
        :return: Area as a float.
        """
        return detection[2] * detection[3]

    def select_target(self, detections, yellow_goal_mode=None):
        """
        Select the optimal target based on detection size and distance to center.
        
        :param detections: YOLO detection results.
        :param yellow_goal_mode: Optional mode to filter detections by goal color.
        :return: A Detection message for the chosen target, or None.
        """
        # No detections available.
        if not detections or detections.boxes.id is None:
            self.frames_without_target += 1
            if self.frames_without_target >= self.lost_track_threshold:
                self.current_tracked_id = None
            return None

        boxes = detections.boxes.xywh.cpu()
        track_ids = detections.boxes.id.int().cpu().tolist()

        best_target = None

        # If already tracking an object, try to continue tracking it.
        if self.current_tracked_id is not None:
            if self.current_tracked_id in track_ids:
                self.frames_without_target = 0
                best_target = self.current_tracked_id
            else:
                self.frames_without_target += 1
                if self.frames_without_target >= self.lost_track_threshold:
                    self.current_tracked_id = None
        else:
            # If not tracking, choose a new target.
            if yellow_goal_mode is not None:
                if yellow_goal_mode:
                    boxes = boxes[detections.boxes.cls.cpu() < self.goal_cutoff_index]
                    track_ids = track_ids[detections.boxes.cls.cpu() < self.goal_cutoff_index]
                else:
                    boxes = boxes[detections.boxes.cls.cpu() >= self.goal_cutoff_index]
                    track_ids = track_ids[detections.boxes.cls.cpu() >= self.goal_cutoff_index]
            else:
                boxes = boxes[detections.boxes.cls.cpu() < 2]
                track_ids = track_ids[detections.boxes.cls.cpu() < 2]
                
            best_score = float('inf')
            for box, track_id in zip(boxes, track_ids):
                center_distance = self.calculate_center_distance(box)
                area = self.calculate_area(box)
                if area < 40:  # Ignore very small detections.
                    continue
                score = (center_distance / area) * 100  # Lower score is better.
                if score < best_score:
                    best_score = score
                    best_target = track_id

        if best_target is None:
            return None

        self.current_tracked_id = best_target
        self.frames_without_target = 0

        # Find the detection corresponding to the tracked target.
        best_box = detections[track_ids.index(self.current_tracked_id)]
        detection_msg = Detection()
        detection_msg.obj_class = detections.names[int(best_box.boxes.cls.cpu().tolist()[0])]
        detection_msg.bbox = best_box.boxes.xywh.cpu().tolist()[0]
        detection_msg.confidence = best_box.boxes.conf.cpu().tolist()[0]
        detection_msg.track_id = self.current_tracked_id

        return detection_msg
