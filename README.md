# VisionModule
This repository houses the code that will directly enables computer vision for the CatchingBlimp system

Certainly! I'll provide you with a `.gitignore` file for ROS 2 colcon package build files and documentation on how to run and use the stereo vision code.

First, let's create a `.gitignore` file:

```
# ROS 2 build and install directories
build/
install/
log/

# Colcon-related files
.colcon_cd_root
.colcon/

# Python-related files
__pycache__/
*.py[cod]

# C++ build artifacts
*.o
*.so
*.a

# IDE-specific files
.vscode/
.idea/

# Temporary files
*.swp
*~

# OS-specific files
.DS_Store
Thumbs.db
```

Now, let's provide documentation on how to run and use the stereo vision code:

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
   ros2 run stereo_vision stereo_vision
   ```

## Configuration

The node uses several parameters that can be configured:

- `camera_index`: Index of the camera device (default: 0)
- `publish_intermediate`: Whether to publish intermediate images (default: true)
- `node_namespace`: Namespace for the node (default: "BurnCreamBlimp")
- `camera_number`: Camera number for calibration files (default: 1)

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
ros2 run stereo_vision stereo_vision --ros-args -p camera_index:=1 -p publish_intermediate:=false
```

## Topics

The node publishes the following topics:

- `<namespace>/left_raw`: Raw left camera image
- `<namespace>/right_raw`: Raw right camera image
- `<namespace>/left_rect`: Rectified left camera image
- `<namespace>/right_rect`: Rectified right camera image
- `<namespace>/disparity`: Disparity image

Performance measurement topics:
- `performance/split_time`
- `performance/debay_time`
- `performance/rectify_time`
- `performance/disparity_time`
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
