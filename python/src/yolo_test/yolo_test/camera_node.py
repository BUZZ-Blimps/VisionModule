
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

from yolo_test.ball_tracker import BallTracker
from yolo_test.contour_goal_detection import contour_find_goal
from yolo_test.blob_detector import BlobDetectorClass

# Initialize GStreamer once at startup.
Gst.init(None)
cv2.setUseOptimized(True)
cv2.setNumThreads(2)
print(f'OpenCV enabled?: {cv2.ocl.haveOpenCL()}')

class CameraNode(Node):
    """ROS2 Node for vision processing using YOLO"""

    def __init__(self):
        super().__init__('yolo_test_node')
        self._declare_parameters()
        self._get_parameters()

        # Initialize capture device.
        if self.input_video_path == "":
            self.cap = cv2.VideoCapture(self.device_path)
            if not self.cap.isOpened():
                self.get_logger().error(f'Failed to open camera at {self.device_path}')
                return
        else:
            self.cap = cv2.VideoCapture(self.input_video_path)
            if not self.cap.isOpened():
                self.get_logger().error(f'Failed to open camera at {self.input_video_path}')
                return

        self._setup_camera()

        # Initialize YOLO models.
        self.ball_rknn = YOLO(self.ball_model_file, task='detect')
        self.goal_rknn = YOLO(self.goal_model_file, task='detect')

        # Other variables.
        self.bridge = CvBridge()
        self.ball_search_mode = True
        self.yellow_goal_mode = False
        self.frame_counter = 0
        self.track_history = defaultdict(list)
        self.tracker = BallTracker(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2,
                                   self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.state = 0

        # Publishers.
        self.pub_performance = self.create_publisher(PerformanceMetrics, 'performance_metrics', 10)
        self.pub_detections = self.create_publisher(Float64MultiArray, 'targets', 10)
        self.pub_grid = self.create_publisher(Float64MultiArray, 'grid_distances', 10)

        # Subscribers.
        bool_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.state_sub = self.create_subscription(Int64MultiArray, 'state', self.state_callback, 10)
        self.goalcolor_sub = self.create_subscription(Bool, 'goal_color', self.goal_color_callback, bool_qos)

        # Create thread pool for parallel processing.
        self.thread_pool = ThreadPoolExecutor(max_workers=2)

        # Create a timer callback at 15 Hz.
        self.timer = self.create_timer(1 / 15.0, self.camera_callback)

        # Set up GStreamer streaming pipeline.
        self._setup_gstreamer()

        # Set up Video Recorder if flag enabled.
        self._setup_videosaver()

    def _declare_parameters(self):
        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_number', 4),
                ('device_path', '/dev/video0'),
                ('ball_model_file', 'models/ball.xml'),
                ('goal_model_file', 'models/goal.xml'),
                ('verbose_mode', False),
                ('save_frames', False),
                ('save_location', 'frames/'),
                ('goal_circle_height', 1.13),
                ('goal_square_height', 1.17),
                ('goal_triangle_height', 1.50),
                ('input_video_path', ""),
            ]
        )

    def _get_parameters(self):
        self.camera_number = self.get_parameter('camera_number').value
        self.device_path = self.get_parameter('device_path').value
        self.ball_model_file = self.get_parameter('ball_model_file').value
        self.goal_model_file = self.get_parameter('goal_model_file').value
        self.verbose_mode = self.get_parameter('verbose_mode').value
        self.save_frames = self.get_parameter('save_frames').value
        self.save_location = self.get_parameter('save_location').value
        self.goal_circle_height = self.get_parameter('goal_circle_height').value
        self.goal_square_height = self.get_parameter('goal_square_height').value
        self.goal_triangle_height = self.get_parameter('goal_triangle_height').value
        self.input_video_path = self.get_parameter('input_video_path').value

    def _setup_camera(self):
        """Configure the camera properties."""
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _setup_gstreamer(self):
        """Configure and launch the GStreamer pipeline for streaming."""
        hostname = socket.gethostname()
        match = re.search(r'(\d+)$', hostname)
        device_num = int(match.group(1)) if match else 0
        port = 5000 + device_num

        pipeline_str = (
            "appsrc name=mysource emit-signals=True is-live=True format=GST_FORMAT_TIME "
            "caps=video/x-raw,format=BGR,width=640,height=480,framerate=12/1 ! "
            "videoconvert ! videoscale ! video/x-raw,format=I420,width=640,height=480 ! "
            "mpph264enc name=hwenc ! "
            "h264parse ! rtph264pay config-interval=1 pt=96 ! "
            "application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96 ! "
            "udpsink host=192.168.0.211 port={port} sync=false"
        ).format(port=port)

        self.get_logger().info("Launching GStreamer pipeline:\n" + pipeline_str)
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsrc = self.pipeline.get_by_name("mysource")
        #caps = Gst.Caps.from_string("video/x-raw,width=640,height=480,framerate=12/1")
        #self.appsrc.set_property("caps", caps)
        self.pipeline.set_state(Gst.State.PLAYING)

        self.stream_timestamp = 0.0
        self.frame_duration = 1.0 / 12.0

    def _setup_videosaver(self):
        if self.save_frames:
            # Ensure the save directory exists.
            os.makedirs(self.save_location, exist_ok=True)
            # Capture an initial frame to determine the frame size.
            ret, frame = self.capture_frame()
            if ret:
                # Left frame dimensions: half the width of the full frame.
                # frame_width = frame.shape[1] // 2
                frame_width = frame.shape[1]
                frame_height = frame.shape[0]
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_path = os.path.join(self.save_location, f'{time.time()}.avi')
                self.video_writer = cv2.VideoWriter(video_path, fourcc, 12, (frame_width, frame_height))
                self.get_logger().info(f'Video recording initialized: {video_path}')
            else:
                self.get_logger().error('Failed to capture initial frame for video writer initialization')

    def capture_frame(self):
        """Capture a frame from the camera."""
        return self.cap.read()

    def state_callback(self, msg):
        old_state = self.state

        self.state = msg.data[0]

        self.ball_search_mode = (self.state in [0, 1, 2, 3])

        if old_state is not self.state and self.ball_search_mode:
            self.get_logger().info('Blimp switched from goal search to ball search, Switching vision model...')
        elif old_state is not self.state and not self.ball_search_mode:
            self.get_logger().info('Blimp switched from ball search to goal search, Switching vision model...')

    def goal_color_callback(self, msg):
        self.yellow_goal_mode = msg.data

    def run_model_ball(self, left_frame):
        """
        Run the YOLO detection model on the provided frame.
        
        :param left_frame: Image from the left camera.
        :return: (inference_time, detection results)
        """
        t_start = time.time()
        
        # Only detect class 0 for the ball model
        results = self.ball_rknn.track(
            left_frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            conf=0.65,
            # conf=0.01,
        )[0]
        
        return time.time() - t_start, results

    def run_model_goal(self, left_frame):
        """
        Run the YOLO detection model on the provided frame.
        
        :param left_frame: Image from the left camera.
        :return: (inference_time, detection results)
        """
        t_start = time.time()
        
        if self.ball_search_mode:
            selected_model = self.ball_rknn
            # Only detect class 0 for the ball model
            results = selected_model.track(
                left_frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                conf=0.65,
                classes=[0]  # Only class 0
            )[0]
        else:
            selected_model = self.goal_rknn
            # Detect all classes for the goal model (do not use 'classes' argument)
            results = selected_model.track(
                left_frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                conf=0.65
            )[0]
        
        return time.time() - t_start, results

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
        
        detection_msg = None

        if self.ball_search_mode:
            detected = False

            # Search for balls using YOLO model
            timing['preprocessing'] = (time.time() - t_preprocess_start) * 1000
            t_yolo, detections = self.run_model_ball(left_frame)

            detection_msg = self.tracker.select_target(
                detections,
                yellow_goal_mode=None if self.ball_search_mode else self.yellow_goal_mode
            )

            timing['yolo_inference'] = t_yolo * 1000

            # Process detections.
            if detection_msg is not None:

                # Turn off disparity and depth for now
                detection_msg.depth = 0.0
                theta_x = 0.0
                theta_y = 0.0

                self.pub_detections.publish(Float64MultiArray(data=[
                    detection_msg.bbox[0],
                    detection_msg.bbox[1],
                    detection_msg.depth,
                    detection_msg.track_id * 1.0,
                    (not self.ball_search_mode) * 1.0,
                    theta_x, theta_y,
                    detection_msg.bbox[2], detection_msg.bbox[3]
                ]))
                detected = True

            if not detected:
                # nothing founded
                timing['preprocessing'] = 0.0
                timing['yolo_inference'] = 0.0
                self.pub_detections.publish(Float64MultiArray(data=[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]))

        # else:
        #     # Search for goals
        #     detected = False
            
        #     # Try using YOLO
        #     timing['preprocessing'] = (time.time() - t_preprocess_start) * 1000
        
        #     #yolo_future = self.thread_pool.submit(self.run_model, left_frame)
        #     t_yolo, detections = self.run_model_goal(left_frame) 

        #     detection_msg = self.tracker.select_target(
        #         detections,
        #         yellow_goal_mode=None if self.ball_search_mode else self.yellow_goal_mode
        #     )

        #     timing['yolo_inference'] = t_yolo * 1000

        #     # Process detections.
        #     if detection_msg is not None:
        #         # Goal detection mode: use the calibrated vertical focal length and real goal height.
        #         # Assume detection_msg.obj_class contains a string like "circle", "square", or "triangle"
        #         obj_class = detection_msg.obj_class.lower()
        #         if "circle" in obj_class:
        #             real_height = self.goal_circle_height
        #         elif "triangle" in obj_class:
        #             real_height = self.goal_triangle_height
        #         elif "square" in obj_class:
        #             real_height = self.goal_square_height
        #         raw_goal_height = (self.goal_vertical_focal * real_height) / detection_msg.bbox[3]
        #         # raw_depth = (self.goal_vertical_focal * real_height) / detection_msg.bbox[3]

        #         # Use the goal_vertical_focal loaded from calibration.
        #         detection_msg.depth = 0.0
        #         theta_x = 0.0
        #         theta_y = 0.0
            
        #         self.pub_detections.publish(Float64MultiArray(data=[
        #             detection_msg.bbox[0],
        #             detection_msg.bbox[1],
        #             detection_msg.depth,
        #             detection_msg.track_id * 1.0,
        #             (not self.ball_search_mode) * 1.0,
        #             theta_x, theta_y,
        #             detection_msg.bbox[2], detection_msg.bbox[3]
        #         ]))

        #         detected = True
            
        #     if not detected:
        #         self.pub_detections.publish(Float64MultiArray(data=[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]))

        # Prepare debug view.
        debug_view = left_frame.copy()
        img_h, img_w = debug_view.shape[:2]
        center_img = (img_w // 2, img_h // 2)
        # Draw horizontal line
        cv2.line(debug_view, (center_img[0] - 10, center_img[1]), (center_img[0] + 10, center_img[1]), (0, 0, 255), 2)
        # Draw vertical line
        cv2.line(debug_view, (center_img[0], center_img[1] - 10), (center_img[0], center_img[1] + 10), (0, 0, 255), 2)

        if detection_msg is not None:
            x, y, w, h = detection_msg.bbox.astype(int)
            # Define the bounding box corners (assuming x,y is the center of the box)
            top_left = (x - w // 2, y - h // 2)
            bottom_right = (x + w // 2, y + h // 2)
            cv2.rectangle(debug_view, top_left, bottom_right, (0, 255, 0), 2)
            
            # Draw a marker (filled circle) at the center of the bounding box
            cv2.circle(debug_view, (x, y), 3, (255, 0, 0), -1)
            
            cv2.putText(debug_view,
                        f'{detection_msg.obj_class} {detection_msg.track_id} {detection_msg.depth:.1f}m',
                        (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Stream the processed frame.
        processed = cv2.resize(debug_view, (640, 480))
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
        # perf_msg.yolo_time = timing['yolo_inference']
        perf_msg.total_time = timing['total']
        perf_msg.fps = (1 / perf_msg.total_time) * 1000
        self.pub_performance.publish(perf_msg)

        if self.save_frames:
            # self.video_writer.write(left_frame)
            self.video_writer.write(frame)


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.save_frames:
            node.video_writer.release()
        node.thread_pool.shutdown(wait=True)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
