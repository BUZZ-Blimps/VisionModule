#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import Bool, Float64MultiArray, Int8MultiArray
from sensor_msgs.msg import CompressedImage
from blimp_vision_msgs.msg import PerformanceMetrics, Detection
import yaml
import os
from collections import deque
import time
from rclpy.executors import MultiThreadedExecutor
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict
from rclpy.callback_groups import ReentrantCallbackGroup
from blimp_vision.ball_tracker import BallTracker

from ultralytics import YOLO

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

        # Initialize OpenCV capture
        self.cap = cv2.VideoCapture(self.device_path)
        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open camera at {self.device_path}')
            return

        # Initialize camera properties and parameters
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Initialize camera calibration for left and right cameras
        self.left_camera_matrix, self.right_camera_matrix = None, None
        self.left_distortion_coefficients, self.right_distortion_coefficients = None, None
        self.load_calibration()

        # Initialize YOLO models
        self.ball_rknn = YOLO(self.ball_model_file,task='detect')
        self.goal_rknn = YOLO(self.goal_model_file,task='detect')

        # Compute rectification maps
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.left_camera_matrix, self.left_distortion_coefficients, None, None, (640, 480), cv2.CV_32FC1)
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            self.right_camera_matrix, self.right_distortion_coefficients, None, None, (640, 480), cv2.CV_32FC1)
        
        # Initialize stereo matcher
        self.stereo = cv2.StereoBM.create(
            numDisparities=64,  # Reduced from 128 for speed
            blockSize=9        # Smaller block size for faster computation
        )
        
        # Additional StereoBM parameters for better speed/quality trade-off
        self.stereo.setPreFilterSize(5)
        self.stereo.setPreFilterCap(61)
        self.stereo.setMinDisparity(0)
        self.stereo.setTextureThreshold(507)
        self.stereo.setUniquenessRatio(0)
        self.stereo.setSpeckleWindowSize(0)
        self.stereo.setSpeckleRange(8)
        self.stereo.setDisp12MaxDiff(1)

        # Initialize other variables
        self.bridge = CvBridge()
        self.ball_search_mode = True
        self.yellow_goal_mode = False
        self.goal_color = None
        self.frame_counter = 0
        self.track_history = defaultdict(lambda: [])
        self.tracker = BallTracker(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2, self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize publishers
        self.pub_performance = self.create_publisher(PerformanceMetrics, 'performance_metrics', 10)
        #self.pub_detections = self.create_publisher(Detection, 'detections', 10)
        self.pub_detections = self.create_publisher(Float64MultiArray, 'targets', 10)
        self.pub_debug_view = self.create_publisher(CompressedImage, 'debug_view', 10)

        # Initialize subscriber
        self.vision_mode_sub = self.create_subscription(Int8MultiArray, 'vision_toggle', self.vision_mode_callback, 10)

        self.camera_lock = threading.Lock()
        self.frame_queue = deque(maxlen=4)
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        self.callback_group = ReentrantCallbackGroup()
        self.timer = self.create_timer(1/60.0, self.camera_callback, callback_group=self.callback_group)

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

    def capture_frame(self):
        with self.camera_lock:
            return self.cap.read()
        
    def vision_mode_callback(self, msg):
        """Handle vision mode changes"""
        self.ball_search_mode = bool(msg.data[0])
        self.yellow_goal_mode = bool(msg.data[1])
        if self.verbose_mode:
            self.get_logger().info(f'Vision mode changed to: {self.vision_mode}')

    def run_model(self, left_square):
        t_yolo = time.time()
        selected_model = self.ball_rknn if self.ball_search_mode else self.goal_rknn
        out = selected_model.track(left_square, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        return time.time() - t_yolo, out

    def compute_disparity(self, left_rectified, right_rectified):
        """Fast disparity computation using StereoBM"""
        t_disp = time.time()
        
        # Convert to grayscale for StereoBM
        left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
        
        # Compute disparity
        disparity = self.stereo.compute(left_gray, right_gray)
        
        # Convert to float and scale (StereoBM returns 16-bit fixed-point)
        disparity = disparity.astype(np.float32) / 16.0
        
        return time.time() - t_disp, disparity
    
    def filter_disparity(self, disparity, bbox):
        """Filter disparity values within bounding box"""
        x, y, w, h = bbox.astype(int)
        disparity_roi = disparity[y-h//2:y+h//2, x-w//2:x+w//2]
        return np.mean(disparity_roi) * 1.0

    def camera_callback(self):
        timing = {}
        t_total, t_preprocess_start = time.time(), time.time()

        ret, frame = self.capture_frame()
        if not ret:
            self.get_logger().error('Failed to capture frame')
            return

        # Preprocess frames
        frame_width = frame.shape[1] // 2
        left_frame = frame[:, :frame_width]
        right_frame = frame[:, frame_width:]

        #yolo_future = self.thread_pool.submit(self.run_model, left_frame)

        # Rectify images
        left_rectified = cv2.remap(left_frame, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_frame, self.right_map1, self.right_map2, cv2.INTER_LINEAR) 
        timing['preprocessing'] = (time.time() - t_preprocess_start) * 1000

        t_disparity, disparity = self.compute_disparity(left_rectified, right_rectified)
        timing['disparity'] = t_disparity * 1000

        # Get results and timing
        #t_yolo, detections = yolo_future.result()
        t_yolo, detections = self.run_model(left_frame)
        timing['yolo_inference'] = t_yolo * 1000

        #Post-process detections
        detection_msg = self.tracker.select_target(detections, yellow_goal_mode = None if self.ball_search_mode else self.yellow_goal_mode)
        
        if detection_msg is not None:
            detection_msg.depth = self.filter_disparity(disparity, detection_msg.bbox)
            #self.pub_detections.publish(detection_msg)
            self.pub_detections.publish(Float64MultiArray(data=[detection_msg.bbox[0], detection_msg.bbox[1], detection_msg.depth]))

        # Publish debug view with drawn on detection, depth, class, and track id
        debug_view = left_frame.copy()
        if detection_msg is not None:
            x, y, w, h = detection_msg.bbox.astype(int)
            cv2.rectangle(debug_view, (x-w//2, y-h//2), (x+w//2, y+h//2), (0, 255, 0), 2)
            cv2.putText(debug_view, f'{detection_msg.obj_class} {detection_msg.track_id} {detection_msg.depth:.1f}m', (x-w//2, y-h//2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        self.pub_debug_view.publish(self.bridge.cv2_to_compressed_imgmsg(debug_view))

        timing['total'] = (time.time() - t_total) * 1000

        # self.get_logger().info(
        #     f'Timing => Preprox={timing["preprocessing"]:.1f}, '
        #     f'YOLO={timing["yolo_inference"]:.1f}, '
        #     f'Disparity={timing["disparity"]:.1f}, '
        #     f'Total={timing["total"]:.1f} | '
        #     f'FPS={1 / timing["total"] * 1000:1f}'
        # )

        # Publish performance metrics
        msg = PerformanceMetrics()
        msg.yolo_time = timing['yolo_inference']
        msg.disparity_time = timing['disparity']
        msg.total_time = timing['total']
        self.pub_performance.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    
    # Use MultiThreadedExecutor for the ROS2 node
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    
    try:
        executor.spin()
    finally:
        node.thread_pool.shutdown()
        node.destroy_node()
        rclpy.shutdown()