#!/usr/bin/env python3
"""
Module defining the CameraNode, which captures images, computes disparity,
runs YOLO detection in parallel, streams the debug view via GStreamer, and
publishes a 3x3 grid of distance values for the entire frame.
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

        if not self.undistort_camera:
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
        # New publisher for the 3x3 grid of distance values.
        self.pub_grid = self.create_publisher(Float64MultiArray, 'grid_distances', 10)

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
                ('device_path', '/dev/video0'),
                ('calibration_path', 'calibration/'),
                ('ball_model_file', 'models/ball.xml'),
                ('goal_model_file', 'models/goal.xml'),
                ('verbose_mode', False),
                ('save_frames', False),
                ('save_location', 'frames/'),
                ('undistort_camera', True)
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
        self.undistort_camera = self.get_parameter('undistort_camera').value

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
        self.stereo = cv2.StereoBM.create(numDisparities=64, blockSize=15)
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
        results = selected_model.track(left_frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=0.5)[0]
        return time.time() - t_start, results

    def compute_disparity(self, left_frame, right_frame):
        t_start = time.time()
        
        if self.undistort_camera:
            # Use rectification maps if undistortion is enabled.
            left_rectified = cv2.remap(left_frame, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
            right_rectified = cv2.remap(right_frame, self.right_map1, self.right_map2, cv2.INTER_LINEAR)
        else:
            # Use the raw frames
            left_rectified = left_frame
            right_rectified = right_frame

        left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
        disparity = self.stereo.compute(left_gray, right_gray)
        disparity = disparity.astype(np.float32) / 16.0
        return time.time() - t_start, disparity


    def compute_depth(self, disparity_value):
        """
        Compute depth using the stereo camera model.
        The result is converted to meters.
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
        Filter disparity values within a bounding box and compute the depth.
        Assumes bbox = [center_x, center_y, width, height].
        """
        x, y, w, h = bbox.astype(int)
        x_min = max(x - w // 2, 0)
        x_max = min(x + w // 2, disparity.shape[1])
        y_min = max(y - h // 2, 0)
        y_max = min(y + h // 2, disparity.shape[0])
        disparity_roi = disparity[y_min:y_max, x_min:x_max]
        if disparity_roi.size == 0:
            return float('inf')
        mean_disp = np.mean(disparity_roi)
        return self.compute_depth(mean_disp)

    def compute_grid_depths(self, disparity):
        """
        Divide the disparity map into a 3x3 grid and compute a representative depth (in meters)
        for each region.
        
        :param disparity: The full disparity map.
        :return: A list of 9 depth values.
        """
        h, w = disparity.shape
        grid_depths = []
        cell_h = h // 3
        cell_w = w // 3
        for i in range(3):
            for j in range(3):
                y_min = i * cell_h
                y_max = (i + 1) * cell_h if i < 2 else h
                x_min = j * cell_w
                x_max = (j + 1) * cell_w if j < 2 else w
                region = disparity[y_min:y_max, x_min:x_max]
                if region.size == 0:
                    depth = float('inf')
                else:
                    mean_disp = np.mean(region)
                    depth = self.compute_depth(mean_disp)
                grid_depths.append(depth)
        return grid_depths

    def camera_callback(self):
        """Main callback: capture, process, stream frames, and publish metrics and grid distances."""
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

        # Run YOLO and compute disparity concurrently.
        yolo_future = self.thread_pool.submit(self.run_model, left_frame)
        t_disp, disparity = self.compute_disparity(left_frame, right_frame)

        # Publish grid depth values.
        grid_depths = self.compute_grid_depths(disparity)
        self.pub_grid.publish(Float64MultiArray(data=grid_depths))

        t_yolo, detections = yolo_future.result()
        timing['disparity'] = t_disp * 1000
        timing['yolo_inference'] = t_yolo * 1000

        # Process detections.
        detection_msg = self.tracker.select_target(
            detections,
            yellow_goal_mode=None if self.ball_search_mode else self.yellow_goal_mode
        )
        if detection_msg is not None:
            detection_msg.depth = self.filter_disparity(disparity, detection_msg.bbox)
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