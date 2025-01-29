#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import Bool, Float64MultiArray
from sensor_msgs.msg import Image
from blimp_vision_msgs.msg import PerformanceMetrics, DetectionArray
import yaml
import time
import os
from collections import deque

class CameraNode(Node):
    def __init__(self):
        super().__init__('blimp_vision_node')
        
        # Declare all parameters with defaults
        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_number', 0),
                ('device_path', '/dev/video1'),
                ('calibration_path', 'calibration/'),
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

        # Initialize camera properties and parameters
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Initialize camera calibration for left and right cameras
        self.left_camera_matrix = None
        self.left_distortion_coefficients = None

        self.right_camera_matrix = None
        self.right_distortion_coefficients = None

        self.load_calibration()

        # FOR PERPLEXITY: INITIALIZE YOLO RKNN MODELS

        # Compute rectification maps
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.left_camera_matrix, self.left_distortion_coefficients, None, None, (640, 480), cv2.CV_32FC1)
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            self.right_camera_matrix, self.right_distortion_coefficients, None, None, (640, 480), cv2.CV_32FC1)

        self.frame_times = deque(maxlen=30)
        self.frame_counter = 0
        
        # Initialize stereo matcher
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=100,
            speckleRange=32
        )

        # Initialize other variables
        self.bridge = CvBridge()
        self.vision_mode = False
        
        # Create timer for camera capture
        self.timer = self.create_timer(0.033, self.camera_callback)  # 30 FPS

    def load_calibration(self):
        """Load camera calibration parameters for left and right cameras"""
        try:
            # Load left camera calibration
            left_file = f'camera{self.camera_number}_elp_left.yaml'
            with open(os.path.join(self.calibration_path, left_file), 'r') as file:
                left_data = yaml.safe_load(file)
                self.left_camera_matrix = np.array(left_data['camera_matrix']['data']).reshape(3,3)
                self.left_dist_coeffs = np.array(left_data['distortion_coefficients']['data'])
                self.left_rect_matrix = np.array(left_data['rectification_matrix']['data']).reshape(3,3)
                self.left_proj_matrix = np.array(left_data['projection_matrix']['data']).reshape(3,4)

            # Load right camera calibration
            right_file = f'camera{self.camera_number}_elp_right.yaml'
            with open(os.path.join(self.calibration_path, right_file), 'r') as file:
                right_data = yaml.safe_load(file)
                self.right_camera_matrix = np.array(right_data['camera_matrix']['data']).reshape(3,3)
                self.right_dist_coeffs = np.array(right_data['distortion_coefficients']['data'])
                self.right_rect_matrix = np.array(right_data['rectification_matrix']['data']).reshape(3,3)
                self.right_proj_matrix = np.array(right_data['projection_matrix']['data']).reshape(3,4)

            self.get_logger().info('Successfully loaded camera calibration files')
        except Exception as e:
            self.get_logger().error(f'Failed to load calibration files: {e}')

    def vision_mode_callback(self, msg):
        """Handle vision mode changes"""
        self.vision_mode = msg.data
        if self.verbose_mode:
            self.get_logger().info(f'Vision mode changed to: {self.vision_mode}')

    def camera_callback(self):
        """Main camera processing loop"""
        start_time = time.time()
        
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to capture frame')
            return

        # Split frame into left and right images
        frame_width = frame.shape[1] // 2
        left_frame = frame[:, :frame_width]
        right_frame = frame[:, frame_width:]

        # Resize frames to 640x480
        left_frame = cv2.resize(left_frame, (640, 480))
        right_frame = cv2.resize(right_frame, (640, 480))

        # Rectify images
        left_rectified = cv2.remap(left_frame, self.left_map1, self.left_map2, 
                                 cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_frame, self.right_map1, self.right_map2, 
                                  cv2.INTER_LINEAR)

        # FOR PERPLEXITY: SEND IMAGES TO VISION MODEL


        # # Convert to grayscale for disparity
        # left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        # right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

        # # Compute disparity map
        # disparity = self.stereo.compute(left_gray, right_gray)
        
        # # Normalize disparity for visualization
        # disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, 
        #                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        if self.vision_mode:
            # Process frame here
            perf_msg = PerformanceMetrics()
            perf_msg.fps = 30.0
            self.pub_performance.publish(perf_msg)
            
            det_msg = DetectionArray()
            self.pub_detections.publish(det_msg)
            
            target_msg = Float64MultiArray()
            target_msg.data = [0.0, 0.0, 0.0]
            self.pub_targets.publish(target_msg)

        # if self.save_frames:
        #     timestamp = self.get_clock().now().nanoseconds
        #     cv2.imwrite(f"{self.save_location}/left_{timestamp}.jpg", left_rectified)
        #     cv2.imwrite(f"{self.save_location}/right_{timestamp}.jpg", right_rectified)
        #     cv2.imwrite(f"{self.save_location}/disparity_{timestamp}.jpg", disparity_normalized)

        # Calculate and store frame processing time
        process_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        self.frame_times.append(process_time)
        self.frame_counter += 1

        # Print performance metrics every 10 frames
        if self.frame_counter % 10 == 0:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            avg_fps = 1000 / avg_time
            self.get_logger().info(f'Average processing time: {avg_time:.2f}ms, FPS: {avg_fps:.2f}')

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
