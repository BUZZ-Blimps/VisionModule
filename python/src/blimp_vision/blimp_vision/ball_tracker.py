import numpy as np


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
    
    def select_target(self, disp_map detections, color_mode = None):
        """Select the optimal target based on size and center proximity"""
        if self.frame_center is None:
            raise ValueError("Frame dimensions not initialized")
        
        if not detections:
            self.frames_without_target += 1
            if self.frames_without_target >= self.lost_track_threshold:
                self.current_tracked_id = None
            return None
        
        boxes = detections.boxes.xywh.cpu()
        track_ids = detections.boxes.id.int().cpu().tolist()   
            
        # If currently tracking an object, look for it first
        if self.current_tracked_id is not None:
            current_target = boxes[track_ids.index(self.current_tracked_id)]
            if current_target is not None:
                self.frames_without_target = 0
                return (current_target, self.current_tracked_id)
            
        # Check if model is in goal mode (yellow = 1, orange = 0)
        if color_mode is not None:
            # Filter out detections that are on either side of the goal_cutoff_index depending on color mode
            if detections.color_mode == 1:
                boxes = boxes[boxes.cls.cpu() >= self.goal_cutoff_index]
            else:
                boxes = boxes[boxes.cls.cpu() < self.goal_cutoff_index]
             
        # Select new target based on size and center proximity
        best_target = None
        best_score = float('inf')
        
        for box, track_id in zip(boxes, track_ids):
            center_distance = self.calculate_center_distance(box)
            area = self.calculate_area(box)
            
            # Score combines distance to center (weighted less) and size (weighted more)
            # Lower score is better
            score = (center_distance / area) * 100
            
            if score < best_score:
                best_score = score
                best_target = (box, track_id)
        
        if best_target is not None:
            self.current_tracked_id = best_target[1]
            self.frames_without_target = 0
        
        return best_target

