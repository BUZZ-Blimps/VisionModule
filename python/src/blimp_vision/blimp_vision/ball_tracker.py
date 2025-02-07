import numpy as np
from blimp_vision_msgs.msg import Detection

class BallTracker:
    def __init__(self, width, height):
        self.current_tracked_id = None
        self.frame_center = (width // 2, height // 2)
        self.lost_track_threshold = 5  # Frames to wait before declaring track lost
        self.frames_without_target = 0
        self.goal_cutoff_index = 3.0
        
    def calculate_center_distance(self, detection):
        """Calculate distance from object center to frame center"""
        return ((detection[0] - self.frame_center[0])**2 + 
                (detection[1] - self.frame_center[1])**2)**0.5
    
    def calculate_area(self, detection):
        """Calculate area of detection bounding box"""
        return detection[2] * detection[3]
    
    def select_target(self, detections, yellow_goal_mode = None):
        """Select the optimal target based on size and center proximity"""
        if self.frame_center is None:
            raise ValueError("Frame dimensions not initialized")
        
        if not detections or detections.boxes.id is None:
            self.frames_without_target += 1
            if self.frames_without_target >= self.lost_track_threshold:
                self.current_tracked_id = None
            return None
        
        boxes = detections.boxes.xywh.cpu()
        track_ids = detections.boxes.id.int().cpu().tolist()   
        
        best_target = None
            
        # If currently tracking an object, look for it first
        if self.current_tracked_id is not None:
            if self.current_tracked_id in track_ids:
                self.frames_without_target = 0
                best_target = self.current_tracked_id
            else:
                self.frames_without_target += 1
                if self.frames_without_target >= self.lost_track_threshold:
                    self.current_tracked_id = None
            
        # No prior detection, look for a new target
        else:
            # Check if model is in goal mode (yellow = 1, orange = 0)
            if yellow_goal_mode is not None:
                # Filter out detections that are on either side of the goal_cutoff_index depending on color mode
                if yellow_goal_mode:
                    boxes = boxes[detections.boxes.cls.cpu() < self.goal_cutoff_index]
                else:
                    boxes = boxes[detections.boxes.cls.cpu() >= self.goal_cutoff_index]
            
            best_score = float('inf')
            for box, track_id in zip(boxes, track_ids):
                center_distance = self.calculate_center_distance(box)
                area = self.calculate_area(box)
                
                if area < 20: # Ignore small detections
                    continue
                
                # Score combines distance to center (weighted less) and size (weighted more)
                # Lower score is better
                score = (center_distance / area) * 100
                
                if score < best_score:
                    best_score = score
                    best_target = track_id
        
        if best_target is None:
            return None

        self.current_tracked_id = best_target
        self.frames_without_target = 0
    
        best_box = detections[track_ids.index(self.current_tracked_id)]

        detections_msg = Detection()
        detections_msg.obj_class = detections.names[int(best_box.boxes.cls.cpu().tolist()[0])]
        detections_msg.bbox = best_box.boxes.xywh.cpu().tolist()[0]
        detections_msg.confidence = best_box.boxes.conf.cpu().tolist()[0]
        detections_msg.track_id = self.current_tracked_id
        
        return detections_msg

