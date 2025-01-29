#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import Bool, Float64MultiArray
from sensor_msgs.msg import Image
from stereo_vision_msgs.msg import PerformanceMetrics, DetectionArray

class CameraNode(Node):
    def __init__(self):
        super().__init__('blimp_vision_node')
        
        # Declare all parameters with defaults
        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_number', 0),
                ('device_path', '/dev/video1'),
                ('calibration_path', 'config/calibration.yaml'),
                ('ball_model_file', 'models/ball.xml'),
                ('goal_model_file', 'models/goal.xml'),
                ('verbose_mode', False),
                ('save_frames', False),
                ('save_location', 'frames/')
            ]
        )
        
        # Get parameters
        self.camera_number = self.get_parameter('camera_number').value
        self.device_path = self.get_parameter('device_path').value
        self.calibration_path = self.get_parameter('calibration_path').value
        self.ball_model_file = self.get_parameter('ball_model_file').value
        self.goal_model_file = self.get_parameter('goal_model_file').value
        self.verbose_mode = self.get_parameter('verbose_mode').value
        self.save_frames = self.get_parameter('save_frames').value
        self.save_location = self.get_parameter('save_location').value

        # Initialize publishers
        self.pub_performance = self.create_publisher(
            PerformanceMetrics, 'performance_metrics', 10)
        self.pub_detections = self.create_publisher(
            DetectionArray, 'detections', 10)
        self.pub_targets = self.create_publisher(
            Float64MultiArray, 'targets', 10)

        # Initialize subscriber
        self.vision_mode_sub = self.create_subscription(
            Bool, 'vision_mode', self.vision_mode_callback, 10)

        # Initialize OpenCV capture
        self.cap = cv2.VideoCapture(self.device_path)
        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open camera at {self.device_path}')
            return

        # Initialize other variables
        self.bridge = CvBridge()
        self.vision_mode = False
        
        # Create timer for camera capture
        self.timer = self.create_timer(0.033, self.camera_callback)  # 30 FPS

    def vision_mode_callback(self, msg):
        """Handle vision mode changes"""
        self.vision_mode = msg.data
        if self.verbose_mode:
            self.get_logger().info(f'Vision mode changed to: {self.vision_mode}')

    def camera_callback(self):
        """Main camera processing loop"""
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to capture frame')
            return

        if self.vision_mode:
            # Process frame here
            # This is where you would add your computer vision processing
            
            # Example performance metrics
            perf_msg = PerformanceMetrics()
            perf_msg.fps = 30.0
            self.pub_performance.publish(perf_msg)
            
            # Example detections
            det_msg = DetectionArray()
            # Fill detection message
            self.pub_detections.publish(det_msg)
            
            # Example targets
            target_msg = Float64MultiArray()
            target_msg.data = [0.0, 0.0, 0.0]  # Example coordinates
            self.pub_targets.publish(target_msg)

        if self.save_frames:
            # Save frame to disk
            cv2.imwrite(f"{self.save_location}/frame_{self.get_clock().now().nanoseconds}.jpg", frame)

    def __del__(self):
        """Cleanup when node is destroyed"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraNode()
    rclpy.spin(camera_node)
    camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
