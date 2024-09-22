#ifndef PUBLISHERS_HPP
#define PUBLISHERS_HPP

#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include "stereo_vision_msgs/msg/performance_metrics.hpp"
#include "stereo_vision_msgs/msg/detection_array.hpp"
#include "stereo_camera_node.hpp"

void publishImages(const StereoCameraNode* node, const cv::Mat& left_raw, const cv::Mat& right_raw, const cv::Mat& left_rect, const cv::Mat& right_rect, const cv::Mat& left_rect_debay, const cv::Mat& right_rect_debay, const stereo_msgs::msg::DisparityImage::SharedPtr& disparity_msg);

void publishImage(const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr& publisher, const cv::Mat& image, const std::string& encoding);

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
    const std::chrono::high_resolution_clock::time_point& end_time);

void publishDetections(const StereoCameraNode* node, const std::vector<stereo_vision_msgs::msg::Detection>& detections);

#endif // PUBLISHERS_HPP