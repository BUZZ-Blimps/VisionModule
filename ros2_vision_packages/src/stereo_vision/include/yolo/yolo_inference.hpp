#ifndef YOLO_INFERENCE_HPP
#define YOLO_INFERENCE_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include "stereo_vision_msgs/msg/detection.hpp"
#include <rknn_api.h>
#include <rclcpp/rclcpp.hpp>
#include "yolo_common.hpp"

bool initRKNN(const std::string& model_path, rknn_app_context_t& app_ctx, rclcpp::Logger logger);
std::vector<stereo_vision_msgs::msg::Detection> performYOLOInference(const cv::Mat& image, rknn_app_context_t& app_ctx, float box_conf_threshold, float nms_threshold, rclcpp::Logger logger);

#endif // YOLO_INFERENCE_HPP