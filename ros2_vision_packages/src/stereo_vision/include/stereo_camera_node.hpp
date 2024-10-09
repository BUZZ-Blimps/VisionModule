#ifndef STEREO_CAMERA_NODE_HPP
#define STEREO_CAMERA_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include "image_processing.hpp"
#include "yolo_inference.hpp"

#include <sensor_msgs/msg/image.hpp>
#include <stereo_msgs/msg/disparity_image.hpp>
#include "stereo_vision_msgs/msg/performance_metrics.hpp"
#include "stereo_vision_msgs/msg/detection.hpp"
#include "stereo_vision_msgs/msg/detection_array.hpp"
#include <std_msgs/msg/float64_multi_array.hpp>

#define NMS_THRESH 0.5f  // Adjust the value as needed
#define BOX_THRESH 0.3f  // Adjust the value as needed

class StereoCameraNode : public rclcpp::Node {
public:
    StereoCameraNode();
    ~StereoCameraNode();

    bool publish_intermediate_;
    cv::Mat P_left_;
    cv::Mat P_right_;
    cv::Ptr<cv::StereoBM> stereo_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_left_raw_, pub_right_raw_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_left_rect_, pub_right_rect_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_left_debay_, pub_right_debay_;
    rclcpp::Publisher<stereo_msgs::msg::DisparityImage>::SharedPtr pub_disparity_;
    rclcpp::Publisher<stereo_vision_msgs::msg::PerformanceMetrics>::SharedPtr pub_performance_;
    rclcpp::Publisher<stereo_vision_msgs::msg::DetectionArray>::SharedPtr pub_detections_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_targets_;

private:
    void timerCallback();
    void processingLoop();

    cv::VideoCapture cap_;
    cv::Mat camera_matrix_left_, camera_matrix_right_, dist_coeffs_left_, dist_coeffs_right_;
    cv::Mat R_left_, R_right_;
    cv::Mat map1_left_, map2_left_, map1_right_, map2_right_;
    cv::Mat Q_;
    cv::Size image_size_;

    // Preallocated matrices
    cv::Mat frame_;
    cv::Mat left_raw_, right_raw_;
    cv::Mat yolo_input_;
    cv::Mat left_debay_, right_debay_;
    cv::Mat left_rect_, right_rect_;
    cv::Mat disparity_;
    cv::Mat disparity_float_;
    cv::Mat left_rect_gray_, right_rect_gray_;

    rclcpp::TimerBase::SharedPtr timer_, param_timer_;

    int camera_index_;
    std::string calibration_file_;
    std::string node_namespace_;
    int camera_number_;
    std::string model_path_;

    std::thread processing_thread_;
    std::mutex frame_mutex_;
    cv::Mat latest_frame_;
    std::atomic<bool> stop_processing_{false};


    rknn_app_context_t rknn_app_ctx;
    std::vector<rknn_tensor_attr> output_attrs_;
    const float nms_threshold      = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;
    
    int model_width_;
    int model_height_;
    int model_channel_;
    float scale_w_;
    float scale_h_;

    std::future<std::vector<stereo_vision_msgs::msg::Detection>> yolo_future_;
};

#endif // STEREO_CAMERA_NODE_HPP