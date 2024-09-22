#ifndef IMAGE_PROCESSING_HPP
#define IMAGE_PROCESSING_HPP

#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include "stereo_vision_msgs/msg/detection.hpp"
#include <stereo_msgs/msg/disparity_image.hpp>
#include "stereo_camera_node.hpp"

class StereoCameraNode;

void updateDisparityMapParams(cv::Ptr<cv::StereoBM>& stereo, const rclcpp::Node* node);
void loadROSCalibration(const std::string& filename, cv::Mat& camera_matrix, cv::Mat& dist_coeffs, cv::Mat& R, cv::Mat& P);
double monoDepthEstimator(double bbox_area);
void estimateDepth(std::vector<stereo_vision_msgs::msg::Detection>& detections, const stereo_msgs::msg::DisparityImage& disparity_msg);
stereo_msgs::msg::DisparityImage::SharedPtr convertToDisparityImageMsg(const cv::Mat& disparity, const StereoCameraNode* node);

#endif // IMAGE_PROCESSING_HPP