#include "publishers.hpp"
#include "stereo_camera_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>

void publishImages(const StereoCameraNode* node, const cv::Mat& left_rect, const cv::Mat& right_rect, const stereo_msgs::msg::DisparityImage::SharedPtr& disparity_msg)
{
    static auto last_publish_time = std::chrono::steady_clock::now();
    auto current_time = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_publish_time).count() >= 100) {
        if (node->publish_intermediate_) {
            publishCompressedImage(node->pub_left_rect_, left_rect);
            publishCompressedImage(node->pub_right_rect_, right_rect);
        }
        node->pub_disparity_->publish(*disparity_msg);
        last_publish_time = current_time;
    }
}

void publishCompressedImage(const rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr& publisher, const cv::Mat& image)
{
    sensor_msgs::msg::CompressedImage compressed_msg;
    compressed_msg.header.stamp = rclcpp::Clock().now();
    compressed_msg.format = "jpeg";
    cv::imencode(".jpg", image, compressed_msg.data);
    publisher->publish(compressed_msg);
}

void publishPerformanceMetrics(
    const StereoCameraNode* node,
    const std::chrono::high_resolution_clock::time_point& split_start,
    const std::chrono::high_resolution_clock::time_point& split_end,
    const std::chrono::high_resolution_clock::time_point& rectify_start,
    const std::chrono::high_resolution_clock::time_point& rectify_end,
    const std::chrono::high_resolution_clock::time_point& disparity_start,
    const std::chrono::high_resolution_clock::time_point& disparity_end,
    const std::chrono::high_resolution_clock::time_point& yolo_start,
    const std::chrono::high_resolution_clock::time_point& yolo_end,
    const std::chrono::high_resolution_clock::time_point& start_time,
    const std::chrono::high_resolution_clock::time_point& end_time)
{
    auto msg = stereo_vision_msgs::msg::PerformanceMetrics();

    msg.split_time = std::chrono::duration_cast<std::chrono::microseconds>(split_end - split_start).count() / 1000.0;
    msg.rectify_time = std::chrono::duration_cast<std::chrono::microseconds>(rectify_end - rectify_start).count() / 1000.0;
    msg.disparity_time = std::chrono::duration_cast<std::chrono::microseconds>(disparity_end - disparity_start).count() / 1000.0;
    msg.yolo_time = std::chrono::duration_cast<std::chrono::microseconds>(yolo_end - yolo_start).count() / 1000.0;
    msg.total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;

    msg.fps = 1000.0 / msg.total_time;  // Convert ms to fps

    node->pub_performance_->publish(msg);
}

void publishDetections(const StereoCameraNode* node, const std::vector<stereo_vision_msgs::msg::Detection>& detections)
{
    stereo_vision_msgs::msg::DetectionArray detection_array_msg;
    detection_array_msg.header.stamp = node->now();
    detection_array_msg.header.frame_id = "camera_frame";
    detection_array_msg.detections = detections;

    node->pub_detections_->publish(detection_array_msg);

    // Initialize variables to store the largest detections
    stereo_vision_msgs::msg::Detection largest_balloon;
    stereo_vision_msgs::msg::Detection largest_orange_goal;
    stereo_vision_msgs::msg::Detection largest_yellow_goal;
    float largest_balloon_area = 0;
    float largest_orange_goal_area = 0;
    float largest_yellow_goal_area = 0;

    // Iterate through detections
    for (const auto& detection : detections) {
        float area = detection.bbox[2] * detection.bbox[3];  // width * height
        TargetClass target_class = static_cast<TargetClass>(detection.class_id);
        
        switch (target_class) {
            case TargetClass::GreenBalloon:
            case TargetClass::PurpleBalloon:
                if (area > largest_balloon_area) {
                    largest_balloon = detection;
                    largest_balloon_area = area;
                }
                break;
            case TargetClass::RedCircle:
            case TargetClass::RedSquare:
            case TargetClass::RedTriangle:
                if (area > largest_orange_goal_area) {
                    largest_orange_goal = detection;
                    largest_orange_goal_area = area;
                }
                break;
            case TargetClass::YellowCircle:
            case TargetClass::YellowSquare:
            case TargetClass::YellowTriangle:
                if (area > largest_yellow_goal_area) {
                    largest_yellow_goal = detection;
                    largest_yellow_goal_area = area;
                }
                break;
            default:
                break;
        }
    }

    // Create and populate targets_msg
    std_msgs::msg::Float64MultiArray targets_msg;
    targets_msg.data.resize(9);
    for (int i = 0; i < 9; i++) {
        targets_msg.data[i] = -1.0;
    }
    
    auto populate_target = [](const stereo_vision_msgs::msg::Detection& detection, std::vector<double>& data, int offset) {
        if (detection.class_id == 0.0 && detection.confidence == 0.0) {
            data[offset] = -1.0;
            data[offset + 1] = -1.0;
            data[offset + 2] = -1.0;
        } else {
            data[offset] = detection.bbox[0] + detection.bbox[2] / 2.0;  // center x
            data[offset + 1] = detection.bbox[1] + detection.bbox[3] / 2.0;  // center y
            data[offset + 2] = detection.depth;
        }
    };

    populate_target(largest_balloon, targets_msg.data, 0);
    populate_target(largest_orange_goal, targets_msg.data, 3);
    populate_target(largest_yellow_goal, targets_msg.data, 6);

    node->pub_targets_->publish(targets_msg);
}
