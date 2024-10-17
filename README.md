# VisionModule
This repository houses the code that will directly enables computer vision for the CatchingBlimp system

# Stereo Vision ROS 2 Package

This package provides stereo vision functionality using ROS 2 and OpenCV.

## Prerequisites

1. ROS 2 (tested on Humble)
2. OpenCV
3. cv_bridge
4. yaml-cpp

## Building the Package

1. Clone the repository into your ROS 2 workspace:
   ```
   cd ~/ros2_ws/src
   git clone <repository_url> stereo_vision
   ```

2. Build the package:
   ```
   cd ~/ros2_ws
   colcon build --packages-select stereo_vision
   ```

3. Source the workspace:
   ```
   source ~/ros2_ws/install/setup.bash
   ```

## Running the Node

1. Make sure your stereo camera is connected and recognized by the system.

2. Launch the stereo vision node:
   ```
   ros2 launch stereo_vision stereo_vision
   ```

## Configuration

The node uses several parameters that can be configured:

- `camera_index`: Index of the camera device (default: 0)
- `publish_intermediate`: Whether to publish intermediate images (default: true)
- `node_namespace`: Namespace for the node (default: "BurnCreamBlimp")
- `camera_number`: Camera number for calibration files (default: 1)
- `model_path`: Path to rknn model weights file (required)

Disparity map parameters:
- `min_disparity`
- `num_disparities`
- `block_size`
- `uniqueness_ratio`
- `speckle_window_size`
- `speckle_range`
- `disp_12_max_diff`

You can set these parameters when launching the node:

```
ros2 launch stereo_vision stereo_vision.launch.py camera_index:=1 publish_intermediate:=false node_namespace:="BurnCreamBlimp" camera_number:=4 model_path:=<path to .rknn file>
```

## Topics

The node publishes the following topics:

- `<namespace>/left_rect/compressed`: Rectified left camera image (compressed)
- `<namespace>/right_rect/compressed`: Rectified right camera image (compressed)
- `<namespace>/disparity`: Disparity image
- `<namespace>/detections`: YOLO Detections message

Performance measurement topics:
- `performance/split_time`
- `performance/debay_time`
- `performance/rectify_time`
- `performance/disparity_time`
- `performance/yolo_time`
- `performance/total_time`
- `performance/total_sum_time`
- `performance/fps`

## Calibration

The node expects calibration files for the left and right cameras in the package's `share` directory. The files should be named:

- `camera<camera_number>_elp_left.yaml`
- `camera<camera_number>_elp_right.yaml`

Ensure these files are present and contain the correct calibration data for your stereo camera setup.

## Troubleshooting

If you encounter issues:

1. Check if the camera is properly connected and recognized.
2. Verify that the calibration files are present and contain valid data.
3. Adjust the disparity map parameters to improve the quality of the disparity image.
4. Monitor the performance topics to identify potential bottlenecks.

For any further issues, please refer to the error messages in the console output or file an issue in the repository.
