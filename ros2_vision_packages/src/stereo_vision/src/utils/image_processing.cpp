#include "image_processing.hpp"
#include <sensor_msgs/image_encodings.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <yaml-cpp/yaml.h>
#include "stereo_camera_node.hpp"

double flength;
double baseline;
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

std::vector<float> removeOutliers(std::vector<float>& disparities, float eps, int minPts) {
    std::vector<bool> visited(disparities.size(), false);
    std::vector<bool> noise(disparities.size(), false);
    std::vector<float> filtered_disparities;

    for (size_t i = 0; i < disparities.size(); ++i) {
        if (visited[i]) continue;
        
        visited[i] = true;
        std::vector<size_t> neighbors;
        
        for (size_t j = 0; j < disparities.size(); ++j) {
            if (std::abs(disparities[i] - disparities[j]) < eps) {
                neighbors.push_back(j);
            }
        }
        
        if (neighbors.size() < minPts) {
            noise[i] = true;
        } else {
            for (size_t j : neighbors) {
                if (!visited[j]) {
                    visited[j] = true;
                    for (size_t k = 0; k < disparities.size(); ++k) {
                        if (std::abs(disparities[j] - disparities[k]) < eps) {
                            neighbors.push_back(k);
                        }
                    }
                }
                if (noise[j]) noise[j] = false;
            }
        }
    }

    for (size_t i = 0; i < disparities.size(); ++i) {
        if (!noise[i]) {
            filtered_disparities.push_back(disparities[i]);
        }
    }

    return filtered_disparities;
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
    cv::Mat disparity;
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(disparity_msg.image, sensor_msgs::image_encodings::TYPE_32FC1);
    } catch (cv_bridge::Exception& e) {
        throw std::runtime_error("cv_bridge exception: " + std::string(e.what()));
    }

    for (auto& detection : detections) {
        try {
            double bbox_area = detection.bbox[2] * detection.bbox[3];
            double mono_depth = monoDepthEstimator(bbox_area);

            std::vector<float> valid_disparities;
            int x1 = detection.bbox[0];
            int y1 = detection.bbox[1];
            int width = detection.bbox[2];
            int height = detection.bbox[3];
            int center_x = x1 + width / 2;
            int center_y = y1 + height / 2;

            // Remove center square
            int square_size = std::min(width, height) / 2;
            int square_x1 = center_x - square_size / 2;
            int square_y1 = center_y - square_size / 2;

            for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++) {
                    int global_x = x1 + i;
                    int global_y = y1 + j;
                    
                    // Skip center square
                    if (global_x >= square_x1 && global_x < square_x1 + square_size &&
                        global_y >= square_y1 && global_y < square_y1 + square_size) {
                        continue;
                    }
                    
                    double disparity_value = cv_ptr->image.at<float>(global_y, global_x);
                    
                    if (!std::isnan(disparity_value) && disparity_value > 0) {
                        valid_disparities.push_back(disparity_value);
                    }
                }
            }

            float depth;
            if (!valid_disparities.empty()) {
                std::sort(valid_disparities.begin(), valid_disparities.end());
                float median = valid_disparities[valid_disparities.size() / 2];
                depth = (disparity_msg.f * disparity_msg.t) / (2*median);

                if (median == 0) {
                    detection.depth = mono_depth;  
                } else if ((depth < mono_depth) || ((detection.class_id != 0) && (detection.class_id != 4))) {
                    detection.depth = depth;
                } else {
                    detection.depth = mono_depth;
                }

                if ((detection.depth > 100.0) || std::isnan(detection.depth)) {
                    detection.depth = 100.0;
                }
            } else {
                detection.depth = mono_depth;
            }

            // Print both depths for testing purposes
            std::cout << "Detection: " << detection.class_id 
                    << ", Mono depth: " << mono_depth 
                    << ", Disparity depth: " << depth 
                    << ", Selected depth: " << detection.depth << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Error in depth estimation: " << e.what() << std::endl;
            std::cerr << "Detection details:" << std::endl;
            std::cerr << "  Class ID: " << detection.class_id << std::endl;
            std::cerr << "  Bounding box: [" << detection.bbox[0] << ", " << detection.bbox[1] 
                    << ", " << detection.bbox[2] << ", " << detection.bbox[3] << "]" << std::endl;
            std::cerr << "  Image dimensions: " << cv_ptr->image.cols << "x" << cv_ptr->image.rows << std::endl;
        }
    }
}

stereo_msgs::msg::DisparityImage::SharedPtr convertToDisparityImageMsg(const cv::Mat& disparity, const StereoCameraNode* node)
{
    auto disparity_msg = std::make_unique<stereo_msgs::msg::DisparityImage>();
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