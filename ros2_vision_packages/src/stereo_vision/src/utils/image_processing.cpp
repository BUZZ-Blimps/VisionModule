#include "image_processing.hpp"
#include <sensor_msgs/image_encodings.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <yaml-cpp/yaml.h>
#include "stereo_camera_node.hpp"

void updateDisparityMapParams(cv::Ptr<cv::StereoBM>& stereo, const rclcpp::Node* node)
{
    stereo->setMinDisparity(node->get_parameter("min_disparity").as_int());
    stereo->setNumDisparities(node->get_parameter("num_disparities").as_int());
    stereo->setBlockSize(node->get_parameter("block_size").as_int());
    stereo->setUniquenessRatio(node->get_parameter("uniqueness_ratio").as_int());
    stereo->setSpeckleWindowSize(node->get_parameter("speckle_window_size").as_int());
    stereo->setSpeckleRange(node->get_parameter("speckle_range").as_int());
    stereo->setDisp12MaxDiff(node->get_parameter("disp_12_max_diff").as_int());
}

void loadROSCalibration(const std::string& filename, cv::Mat& K, cv::Mat& D, cv::Mat& R, cv::Mat& P)
{
    try {
        YAML::Node config = YAML::LoadFile(filename);

        std::vector<double> K_data = config["camera_matrix"]["data"].as<std::vector<double>>();
        std::vector<double> D_data = config["distortion_coefficients"]["data"].as<std::vector<double>>();
        std::vector<double> R_data = config["rectification_matrix"]["data"].as<std::vector<double>>();
        std::vector<double> P_data = config["projection_matrix"]["data"].as<std::vector<double>>();

        K = cv::Mat(3, 3, CV_64F, K_data.data()).clone();
        D = cv::Mat(1, D_data.size(), CV_64F, D_data.data()).clone();
        R = cv::Mat(3, 3, CV_64F, R_data.data()).clone();
        P = cv::Mat(3, 4, CV_64F, P_data.data()).clone();

        std::cout << "Image size: " << cv::Size(config["image_width"].as<int>(), config["image_height"].as<int>()) << std::endl;

        std::cout << "Loaded calibration from " << filename << std::endl;
        std::cout << "K size: " << K.size() << ", type: " << K.type() << std::endl;
        std::cout << "D size: " << D.size() << ", type: " << D.type() << std::endl;
        std::cout << "R size: " << R.size() << ", type: " << R.type() << std::endl;
        std::cout << "P size: " << P.size() << ", type: " << P.type() << std::endl;
    } catch (const YAML::Exception& e) {
        std::cerr << "YAML exception: " << e.what() << " in file: " << filename << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << " in file: " << filename << std::endl;
        throw;
    }
}

double monoDepthEstimator(double bbox_area)
{
    // Model coefficients
    const double a = 2.329;
    const double b = -0.0001485;
    const double c = 0.8616;
    const double d = -0.8479e-05;

    return a * std::exp(b * bbox_area) + c * std::exp(d * bbox_area);
}

void estimateDepth(std::vector<stereo_vision_msgs::msg::Detection>& detections, const stereo_msgs::msg::DisparityImage& disparity_msg)
{
    cv::Mat_<float> depth(disparity_msg.image.height, disparity_msg.image.width);
    float f = disparity_msg.f;
    float T = disparity_msg.t;
    const float* disp_data = reinterpret_cast<const float*>(&disparity_msg.image.data[0]);
    
    for (int y = 0; y < depth.rows; ++y) {
        for (int x = 0; x < depth.cols; ++x) {
            float disparity = disp_data[y * depth.cols + x];
            depth(y, x) = (disparity > disparity_msg.min_disparity) ? (f * T / disparity) : 0.0f;
        }
    }

    for (auto& detection : detections)
    {
        // Mono depth estimation
        double bbox_area = detection.bbox[2] * detection.bbox[3]; // width * height
        double mono_depth = monoDepthEstimator(bbox_area);

        // Disparity-based depth estimation
        int x = static_cast<int>(detection.bbox[0] + detection.bbox[2] / 2);
        int y = static_cast<int>(detection.bbox[1] + detection.bbox[3] / 2);
        float disparity_depth = depth(y, x);

        // Select the minimum of the two depth estimates
        detection.depth = std::min(static_cast<float>(mono_depth), disparity_depth);

        // Print both depths for testing purposes
        std::cout << "Detection: " << detection.label 
                  << ", Mono depth: " << mono_depth 
                  << ", Disparity depth: " << disparity_depth 
                  << ", Selected depth: " << detection.depth << std::endl;
    }
    return;
}

stereo_msgs::msg::DisparityImage::SharedPtr convertToDisparityImageMsg(const cv::Mat& disparity, const StereoCameraNode* node)
{
    auto disparity_msg = std::make_unique<stereo_msgs::msg::DisparityImage>();
    disparity_msg->header.stamp = node->now();
    disparity_msg->header.frame_id = "stereo_camera";

    // Convert disparity to float
    cv::Mat disparity_float;
    disparity.convertTo(disparity_float, CV_32F, 1.0 / 16.0);  // StereoBM computes disparity scaled by 16

    disparity_msg->image = *cv_bridge::CvImage(std_msgs::msg::Header(), sensor_msgs::image_encodings::TYPE_32FC1, disparity_float).toImageMsg();
    disparity_msg->f = node->P_left_.at<double>(0, 0);  // Focal length
    disparity_msg->t = -node->P_right_.at<double>(0, 3) / node->P_right_.at<double>(0, 0);  // Baseline
    disparity_msg->min_disparity = static_cast<float>(node->stereo_->getMinDisparity());
    disparity_msg->max_disparity = static_cast<float>(node->stereo_->getMinDisparity() + node->stereo_->getNumDisparities() - 1);
    
    disparity_msg->image.header = disparity_msg->header;
    disparity_msg->image.height = disparity.rows;
    disparity_msg->image.width = disparity.cols;
    //disparity_msg->image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    disparity_msg->image.step = disparity.cols * sizeof(float);
    //disparity_msg->image.data.assign(reinterpret_cast<const uint8_t*>(disparity.data),
    //                                 reinterpret_cast<const uint8_t*>(disparity.data) + disparity_msg->image.step * disparity.rows);

    return disparity_msg;
}