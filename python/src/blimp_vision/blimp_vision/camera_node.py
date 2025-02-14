#!/usr/bin/env python3
"""
Module defining the CameraNode, which captures images, computes disparity,
runs YOLO detection in parallel, and streams the debug view via GStreamer.
"""

import os
import re
import socket
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import yaml
from cv_bridge import CvBridge
from gi.repository import Gst, GLib
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSHistoryPolicy,
                       QoSReliabilityPolicy, QoSDurabilityPolicy)
from std_msgs.msg import Bool, Float64MultiArray, Int64MultiArray
from blimp_vision_msgs.msg import PerformanceMetrics
from ultralytics import YOLO

from blimp_vision.ball_tracker import BallTracker

# Initialize GStreamer once at startup.
Gst.init(None)


class CameraNode(Node):
    """ROS2 Node for vision processing using YOLO and stereo disparity."""

    def __init__(self):
        super().__init__('blimp_vision_node')
        self._declare_parameters()
        self._get_parameters()

        # Initialize capture device.
        self.cap = cv2.VideoCapture(self.device_path)
        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open camera at {self.device_path}')
            return

        self._setup_camera()
        self._load_calibration()

        # Initialize YOLO models.
        self.ball_rknn = YOLO(self.ball_model_file, task='detect')
        self.goal_rknn = YOLO(self.goal_model_file, task='detect')

        # Compute rectification maps.
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.left_camera_matrix,
            self.left_distortion_coefficients,
            None,
            None,
            (640, 480),
            cv2.CV_32FC1)
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            self.right_camera_matrix,
            self.right_distortion_coefficients,
            None,
            None,
            (640, 480),
            cv2.CV_32FC1)

        # Initialize stereo matcher.
        self._setup_stereo_matcher()

        # Other variables.
        self.bridge = CvBridge()
        self.ball_search_mode = True
        self.yellow_goal_mode = False
        self.frame_counter = 0
        self.track_history = defaultdict(list)
        self.tracker = BallTracker(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2,
                                   self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Publishers.
        self.pub_performance = self.create_publisher(PerformanceMetrics, 'performance_metrics', 10)
        self.pub_detections = self.create_publisher(Float64MultiArray, 'targets', 10)

        # Subscribers.
        bool_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.state_sub = self.create_subscription(Int64MultiArray, 'state', self.state_callback, 10)
        self.goalcolor_sub = self.create_subscription(Bool, 'goal_color', self.goal_color_callback, bool_qos)

        # Create thread pool for parallel processing.
        self.thread_pool = ThreadPoolExecutor(max_workers=2)

        # Create a timer callback at 15 Hz.
        self.timer = self.create_timer(1 / 15.0, self.camera_callback)

        # Set up GStreamer streaming pipeline.
        self._setup_gstreamer()

    def _declare_parameters(self):
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

    def _get_parameters(self):
        self.camera_number = self.get_parameter('camera_number').value
        self.device_path = self.get_parameter('device_path').value
        self.calibration_path = self.get_parameter('calibration_path').value
        self.ball_model_file = self.get_parameter('ball_model_file').value
        self.goal_model_file = self.get_parameter('goal_model_file').value
        self.verbose_mode = self.get_parameter('verbose_mode').value
        self.save_frames = self.get_parameter('save_frames').value
        self.save_location = self.get_parameter('save_location').value

    def _setup_camera(self):
        """Configure the camera properties."""
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _load_calibration(self):
        """Load camera calibration parameters for left and right cameras."""
        try:
            # Left camera calibration.
            left_file = f'camera{self.camera_number}_elp_left.yaml'
            left_path = os.path.join(self.calibration_path, left_file)
            with open(left_path, 'r') as file:
                left_data = yaml.safe_load(file)
                self.left_camera_matrix = np.array(left_data['camera_matrix']['data']).reshape(3, 3)
                self.left_distortion_coefficients = np.array(left_data['distortion_coefficients']['data'])
                self.left_rect_matrix = np.array(left_data['rectification_matrix']['data']).reshape(3, 3)
                self.left_proj_matrix = np.array(left_data['projection_matrix']['data']).reshape(3, 4)

            # Right camera calibration.
            right_file = f'camera{self.camera_number}_elp_right.yaml'
            right_path = os.path.join(self.calibration_path, right_file)
            with open(right_path, 'r') as file:
                right_data = yaml.safe_load(file)
                self.right_camera_matrix = np.array(right_data['camera_matrix']['data']).reshape(3, 3)
                self.right_distortion_coefficients = np.array(right_data['distortion_coefficients']['data'])
                self.right_rect_matrix = np.array(right_data['rectification_matrix']['data']).reshape(3, 3)
                self.right_proj_matrix = np.array(right_data['projection_matrix']['data']).reshape(3, 4)

            self.get_logger().info('Successfully loaded camera calibration files')
        except Exception as e:
            self.get_logger().error(f'Failed to load calibration files: {e}')

    def _setup_stereo_matcher(self):
        """Initialize and configure the stereo matcher."""
        self.stereo = cv2.StereoBM.create(numDisparities=64, blockSize=9)
        self.stereo.setPreFilterSize(5)
        self.stereo.setPreFilterCap(61)
        self.stereo.setMinDisparity(0)
        self.stereo.setNumDisparities(64)
        self.stereo.setBlockSize(5)
        self.stereo.setTextureThreshold(507)
        self.stereo.setUniquenessRatio(5)
        self.stereo.setSpeckleWindowSize(100)
        self.stereo.setSpeckleRange(32)
        self.stereo.setDisp12MaxDiff(1)

    def _setup_gstreamer(self):
        """Configure and launch the GStreamer pipeline for streaming."""
        hostname = socket.gethostname()
        match = re.search(r'(\d+)$', hostname)
        device_num = int(match.group(1)) if match else 0
        port = 5000 + device_num

        pipeline_str = (
            "appsrc name=mysource is-live=true block=true format=time "
            "! videoconvert "
            "! videoscale "
            "! video/x-raw,format=I420,width=320,height=240 "
            "! mpph264enc name=hwenc "
            "! h264parse config-interval=1 "
            "! mpegtsmux "
            "! udpsink host=192.168.0.200 port={port}"
        ).format(port=port)

        self.get_logger().info("Launching GStreamer pipeline:\n" + pipeline_str)
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsrc = self.pipeline.get_by_name("mysource")
        caps = Gst.Caps.from_string("video/x-raw,format=BGR,width=320,height=240,framerate=12/1")
        self.appsrc.set_property("caps", caps)
        self.pipeline.set_state(Gst.State.PLAYING)

        self.stream_timestamp = 0.0
        self.frame_duration = 1.0 / 12.0

    def capture_frame(self):
        """Capture a frame from the camera."""
        return self.cap.read()

    def state_callback(self, msg):
        self.state = msg.data[0]
        self.ball_search_mode = (self.state == 0)

    def goal_color_callback(self, msg):
        self.yellow_goal_mode = msg.data

    def run_model(self, left_frame):
        """
        Run the YOLO detection model on the provided frame.
        
        :param left_frame: Image from the left camera.
        :return: (inference_time, detection results)
        """
        t_start = time.time()
        selected_model = self.ball_rknn if self.ball_search_mode else self.goal_rknn
        results = selected_model.track(left_frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        return time.time() - t_start, results

    def compute_disparity(self, left_frame, right_frame):
        """
        Compute disparity between rectified images using StereoBM.
        
        :return: (computation_time, disparity map)
        """

        t_start = time.time()
        
        # Rectify images.
        left_rectified = cv2.remap(left_frame, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_frame, self.right_map1, self.right_map2, cv2.INTER_LINEAR)
        
        left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
        disparity = self.stereo.compute(left_gray, right_gray)
        disparity = disparity.astype(np.float32) / 16.0
        return time.time() - t_start, disparity

    def compute_depth(self, disparity_value):
        """
        Compute depth using the stereo camera model.
        """
        focal_length = self.left_proj_matrix[0, 0]
        baseline = -self.right_proj_matrix[0, 3] / focal_length
        disparity_value = max(disparity_value, 1e-6)
        depth = ((focal_length * baseline) / disparity_value) / 1000.0
        return depth

    def mono_depth_estimator(self, h, w):
        """
        Estimate depth based on bounding box dimensions using a predefined exponential model.
        """
        a = 2.329
        b = -0.0001485
        c = 0.8616
        d = -0.8479e-05
        bbox_area = h * w
        return a * np.exp(b * bbox_area) + c * np.exp(d * bbox_area)

    def filter_disparity(self, disparity, bbox):
        """
        Filter disparity values within a bounding box and compute depth.
        """
        x, y, w, h = bbox.astype(int)
        disparity_roi = disparity[y - h // 2:y + h // 2, x - w // 2:x + w // 2]
        mean_disp = np.mean(disparity_roi)
        return self.compute_depth(mean_disp)

    def camera_callback(self):
        """Main callback: capture, process, and stream frames while publishing metrics."""
        timing = {}
        t_total = time.time()
        t_preprocess_start = time.time()

        ret, frame = self.capture_frame()
        if not ret:
            self.get_logger().error('Failed to capture frame')
            return

        # Split frame into left and right images.
        frame_width = frame.shape[1] // 2
        left_frame = frame[:, :frame_width]
        right_frame = frame[:, frame_width:]

        timing['preprocessing'] = (time.time() - t_preprocess_start) * 1000

        # Run YOLO and disparity computations concurrently.
        #disparity_future = self.thread_pool.submit(self.compute_disparity, left_frame, right_frame)
        yolo_future = self.thread_pool.submit(self.run_model, left_frame)
        #t_yolo, detections = self.run_model(left_frame)

        t_disp, disparity = self.compute_disparity(left_frame, right_frame)
        #t_disp, disparity = disparity_future.result()
        t_yolo, detections = yolo_future.result()
        timing['disparity'] = t_disp * 1000
        timing['yolo_inference'] = t_yolo * 1000

        # Process detections.
        detection_msg = self.tracker.select_target(
            detections,
            yellow_goal_mode=None if self.ball_search_mode else self.yellow_goal_mode
        )
        if detection_msg is not None:
            # Here you could choose between mono_depth_estimator or filter_disparity.
            detection_msg.depth = self.mono_depth_estimator(detection_msg.bbox[2], detection_msg.bbox[3])
            self.pub_detections.publish(Float64MultiArray(data=[
                detection_msg.bbox[0],
                detection_msg.bbox[1],
                detection_msg.depth
            ]))

        # Prepare debug view.
        debug_view = left_frame.copy()
        if detection_msg is not None:
            x, y, w, h = detection_msg.bbox.astype(int)
            cv2.rectangle(debug_view, (x - w // 2, y - h // 2),
                          (x + w // 2, y + h // 2), (0, 255, 0), 2)
            cv2.putText(debug_view,
                        f'{detection_msg.obj_class} {detection_msg.track_id} {detection_msg.depth:.1f}m',
                        (x - w // 2, y - h // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Stream the processed frame.
        processed = cv2.resize(debug_view, (320, 240))
        data = processed.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.pts = int(self.stream_timestamp * Gst.SECOND)
        buf.duration = int(self.frame_duration * Gst.SECOND)
        self.stream_timestamp += self.frame_duration

        flow_return = self.appsrc.emit("push-buffer", buf)
        if flow_return != Gst.FlowReturn.OK:
            self.get_logger().error("Error pushing buffer: " + str(flow_return))

        timing['total'] = (time.time() - t_total) * 1000

        # Publish performance metrics.
        perf_msg = PerformanceMetrics()
        perf_msg.yolo_time = timing['yolo_inference']
        perf_msg.disparity_time = timing['disparity']
        perf_msg.total_time = timing['total']
        perf_msg.fps = (1 / perf_msg.total_time) * 1000
        self.pub_performance.publish(perf_msg)


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.thread_pool.shutdown(wait=True)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
