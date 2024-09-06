// stereo_camera_node.cpp

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <stereo_msgs/msg/disparity_image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <std_msgs/msg/float64.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <thread>
#include <mutex>
#include <atomic>
#include "stereo_vision_msgs/msg/performance_metrics.hpp"

class StereoCameraNode : public rclcpp::Node
{
public:
    StereoCameraNode() : Node("stereo_camera_node")
    {
        // Declare parameters
        this->declare_parameter("camera_index", 0);
        this->declare_parameter("calibration_file", "");
        this->declare_parameter("publish_intermediate", true);
        this->declare_parameter("node_namespace", "BurnCreamBlimp");
        this->declare_parameter("camera_number", 1);

        // Disparity map parameters
        this->declare_parameter("min_disparity", 0);
        this->declare_parameter("num_disparities", 64);
        this->declare_parameter("block_size", 5);
        this->declare_parameter("uniqueness_ratio", 5);
        this->declare_parameter("speckle_window_size", 100);
        this->declare_parameter("speckle_range", 32);
        this->declare_parameter("disp_12_max_diff", 1);

        // Get parameters
        camera_index_ = this->get_parameter("camera_index").as_int();
        calibration_file_ = this->get_parameter("calibration_file").as_string();
        publish_intermediate_ = this->get_parameter("publish_intermediate").as_bool();
        node_namespace_ = this->get_parameter("node_namespace").as_string();
        camera_number_ = this->get_parameter("camera_number").as_int();

        // Initialize camera
        cap_.open(camera_index_, cv::CAP_V4L2);
        if (!cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open camera");
            return;
        }

        // Set expected resolution
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, 2560);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 960);

        // Verify the settings
        double width = cap_.get(cv::CAP_PROP_FRAME_WIDTH);
        double height = cap_.get(cv::CAP_PROP_FRAME_HEIGHT);
        RCLCPP_INFO(this->get_logger(), "Camera resolution set to: %.0f x %.0f", width, height);

        // Preallocate matrices
        frame_ = cv::Mat(height, width, CV_8UC1);
        left_raw_ = cv::Mat(height, width/2, CV_8UC1);
        right_raw_ = cv::Mat(height, width/2, CV_8UC1);
        left_debay_ = cv::Mat(height, width/2, CV_8UC1);
        right_debay_ = cv::Mat(height, width/2, CV_8UC1);
        left_rect_ = cv::Mat(height, width/2, CV_8UC1);
        right_rect_ = cv::Mat(height, width/2, CV_8UC1);
        disparity_ = cv::Mat(height, width/2, CV_16S);
        disparity_float_ = cv::Mat(height, width/2, CV_32F);

        // Load calibration files
        std::string calib_dir = ament_index_cpp::get_package_share_directory("stereo_vision") + "/calibration/";
        std::string left_calib_file = calib_dir + "camera" + std::to_string(camera_number_) + "_elp_left.yaml";
        std::string right_calib_file = calib_dir + "camera" + std::to_string(camera_number_) + "_elp_right.yaml";

        std::cout << "Left calibration file path: " << left_calib_file << std::endl;
        std::cout << "Right calibration file path: " << right_calib_file << std::endl;

        loadROSCalibration(left_calib_file, camera_matrix_left_, dist_coeffs_left_, R_left_, P_left_);
        loadROSCalibration(right_calib_file, camera_matrix_right_, dist_coeffs_right_, R_right_, P_right_);

        std::cout << "Loaded Calibration files." << std::endl;

        // Compute rectification maps (only once)
        cv::Size image_size(1280, 960);  // Adjust these values if your individual camera image size is different

        // Compute rectification maps for left camera
        cv::initUndistortRectifyMap(camera_matrix_left_, dist_coeffs_left_, R_left_, P_left_,
                                    image_size, CV_32FC1, map1_left_, map2_left_);

        // Compute rectification maps for right camera
        cv::initUndistortRectifyMap(camera_matrix_right_, dist_coeffs_right_, R_right_, P_right_,
                                    image_size, CV_32FC1, map1_right_, map2_right_);

        // Store Q matrix for potential later use (e.g., reprojecting disparity to 3D)
        Q_ = P_right_.colRange(0, 3).inv() * P_left_;

        // Debug output
        RCLCPP_INFO(this->get_logger(), "Rectification maps computed successfully");

        // Initialize StereoBM
        stereo_ = cv::StereoBM::create();
        updateDisparityMapParams();

        // Use default QoS for image topics
        const auto qos = rclcpp::QoS(10);  // Keep last 10 messages

        // Create publishers with default QoS
        if (publish_intermediate_) {
            pub_left_raw_ = this->create_publisher<sensor_msgs::msg::Image>(node_namespace_ + "/left_raw", qos);
            pub_right_raw_ = this->create_publisher<sensor_msgs::msg::Image>(node_namespace_ + "/right_raw", qos);
            pub_left_rect_ = this->create_publisher<sensor_msgs::msg::Image>(node_namespace_ + "/left_rect", qos);
            pub_right_rect_ = this->create_publisher<sensor_msgs::msg::Image>(node_namespace_ + "/right_rect", qos);
        }

        // Use default QoS for disparity publisher
        pub_disparity_ = this->create_publisher<stereo_msgs::msg::DisparityImage>(node_namespace_ + "/disparity", qos);

        // Create performance publisher
        pub_performance_ = this->create_publisher<stereo_vision_msgs::msg::PerformanceMetrics>(
            node_namespace_ + "/performance", 10);

        // Create timer for main loop
        timer_ = this->create_wall_timer(std::chrono::milliseconds(33), std::bind(&StereoCameraNode::timerCallback, this));

        // Create timer for updating disparity map parameters
        param_timer_ = this->create_wall_timer(std::chrono::seconds(1), std::bind(&StereoCameraNode::updateDisparityMapParams, this));

        // Initialize the processing thread
        processing_thread_ = std::thread(&StereoCameraNode::processingLoop, this);
    }

    ~StereoCameraNode()
    {
        stop_processing_ = true;
        if (processing_thread_.joinable()) {
            processing_thread_.join();
        }
    }

private:
    void timerCallback()
    {
        cv::Mat frame;
        cap_ >> frame;

        if (frame.empty()) {
            RCLCPP_WARN(this->get_logger(), "Empty frame");
            return;
        }

        std::lock_guard<std::mutex> lock(frame_mutex_);
        latest_frame_ = frame;
    }

    void processingLoop()
    {
        while (rclcpp::ok() && !stop_processing_) {
            {
                std::lock_guard<std::mutex> lock(frame_mutex_);
                cap_ >> frame_;
                if (frame_.empty()) {
                    RCLCPP_WARN(this->get_logger(), "Empty frame");
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }
            }

            auto start_time = std::chrono::high_resolution_clock::now();

            // Split frame into left and right images
            auto split_start = std::chrono::high_resolution_clock::now();
            left_raw_ = frame_(cv::Rect(0, 0, frame_.cols/2, frame_.rows));
            right_raw_ = frame_(cv::Rect(frame_.cols/2, 0, frame_.cols/2, frame_.rows));
            auto split_end = std::chrono::high_resolution_clock::now();

            // Debayer and convert to grayscale
            auto debay_start = std::chrono::high_resolution_clock::now();
            try {
                cv::cvtColor(left_raw_, left_debay_, cv::COLOR_BGR2GRAY);
                cv::cvtColor(right_raw_, right_debay_, cv::COLOR_BGR2GRAY);

            } catch (const cv::Exception& e) {
                RCLCPP_ERROR(this->get_logger(), "OpenCV exception in debayer: %s", e.what());
                continue;
            }
            auto debay_end = std::chrono::high_resolution_clock::now();

            // Rectify
            auto rectify_start = std::chrono::high_resolution_clock::now();
            cv::remap(left_debay_, left_rect_, map1_left_, map2_left_, cv::INTER_LINEAR);
            cv::remap(right_debay_, right_rect_, map1_right_, map2_right_, cv::INTER_LINEAR);
            auto rectify_end = std::chrono::high_resolution_clock::now();

            // Compute disparity
            auto disparity_start = std::chrono::high_resolution_clock::now();
            stereo_->compute(left_rect_, right_rect_, disparity_);
            auto disparity_end = std::chrono::high_resolution_clock::now();

            auto end_time = std::chrono::high_resolution_clock::now();

            // Publish images with rate limiting
            static auto last_publish_time = std::chrono::steady_clock::now();
            auto current_time = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_publish_time).count() >= 100) {
                if (publish_intermediate_) {
                    publishImage(left_raw_, pub_left_raw_, sensor_msgs::image_encodings::MONO8);
                    publishImage(right_raw_, pub_right_raw_, sensor_msgs::image_encodings::MONO8);
                    publishImage(left_rect_, pub_left_rect_, sensor_msgs::image_encodings::MONO8);
                    publishImage(right_rect_, pub_right_rect_, sensor_msgs::image_encodings::MONO8);
                }
                publishDisparityImage(disparity_);
                last_publish_time = current_time;
            }

            // Publish performance measurements
            publishPerformanceMetrics(
                split_start, split_end,
                debay_start, debay_end,
                rectify_start, rectify_end,
                disparity_start, disparity_end,
                start_time, end_time
            );
        }
    }

    void updateDisparityMapParams()
    {
        stereo_->setMinDisparity(this->get_parameter("min_disparity").as_int());
        stereo_->setNumDisparities(this->get_parameter("num_disparities").as_int());
        stereo_->setBlockSize(this->get_parameter("block_size").as_int());
        stereo_->setUniquenessRatio(this->get_parameter("uniqueness_ratio").as_int());
        stereo_->setSpeckleWindowSize(this->get_parameter("speckle_window_size").as_int());
        stereo_->setSpeckleRange(this->get_parameter("speckle_range").as_int());
        stereo_->setDisp12MaxDiff(this->get_parameter("disp_12_max_diff").as_int());
    }

    void publishImage(const cv::Mat& image, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher, const std::string& encoding)
    {
        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), encoding, image).toImageMsg();
        publisher->publish(*msg);
    }

    void publishDisparityImage(const cv::Mat& disparity)
    {
        auto msg = std::make_unique<stereo_msgs::msg::DisparityImage>();
        msg->header.stamp = this->now();
        msg->header.frame_id = "stereo_camera";

        // Convert disparity to float
        disparity.convertTo(disparity_float_, CV_32F, 1.0 / 16.0);  // StereoBM computes disparity scaled by 16

        msg->image = *cv_bridge::CvImage(std_msgs::msg::Header(), sensor_msgs::image_encodings::TYPE_32FC1, disparity_float_).toImageMsg();
        msg->f = P_left_.at<double>(0, 0);  // Focal length
        msg->t = -P_right_.at<double>(0, 3) / P_right_.at<double>(0, 0);  // Baseline
        msg->min_disparity = static_cast<float>(stereo_->getMinDisparity());
        msg->max_disparity = static_cast<float>(stereo_->getMinDisparity() + stereo_->getNumDisparities() - 1);

        pub_disparity_->publish(std::move(msg));
    }

    void publishPerformanceMetrics(
        const std::chrono::high_resolution_clock::time_point& split_start,
        const std::chrono::high_resolution_clock::time_point& split_end,
        const std::chrono::high_resolution_clock::time_point& debay_start,
        const std::chrono::high_resolution_clock::time_point& debay_end,
        const std::chrono::high_resolution_clock::time_point& rectify_start,
        const std::chrono::high_resolution_clock::time_point& rectify_end,
        const std::chrono::high_resolution_clock::time_point& disparity_start,
        const std::chrono::high_resolution_clock::time_point& disparity_end,
        const std::chrono::high_resolution_clock::time_point& start_time,
        const std::chrono::high_resolution_clock::time_point& end_time)
    {
        auto msg = stereo_vision_msgs::msg::PerformanceMetrics();

        msg.split_time = std::chrono::duration_cast<std::chrono::microseconds>(split_end - split_start).count() / 1000.0;
        msg.debay_time = std::chrono::duration_cast<std::chrono::microseconds>(debay_end - debay_start).count() / 1000.0;
        msg.rectify_time = std::chrono::duration_cast<std::chrono::microseconds>(rectify_end - rectify_start).count() / 1000.0;
        msg.disparity_time = std::chrono::duration_cast<std::chrono::microseconds>(disparity_end - disparity_start).count() / 1000.0;
        msg.total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;

        msg.fps = 1000.0 / msg.total_time;  // Convert ms to fps

        pub_performance_->publish(msg);
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

            if (image_size_.empty()) {
                image_size_ = cv::Size(config["image_width"].as<int>(), config["image_height"].as<int>());
            }

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

    cv::VideoCapture cap_;
    cv::Ptr<cv::StereoBM> stereo_;
    cv::Mat camera_matrix_left_, camera_matrix_right_, dist_coeffs_left_, dist_coeffs_right_;
    cv::Mat R_left_, R_right_, P_left_, P_right_;
    cv::Mat map1_left_, map2_left_, map1_right_, map2_right_;
    cv::Mat Q_;
    cv::Size image_size_;

    // Preallocated matrices
    cv::Mat frame_;
    cv::Mat left_raw_, right_raw_;
    cv::Mat left_debay_, right_debay_;
    cv::Mat left_rect_, right_rect_;
    cv::Mat disparity_;
    cv::Mat disparity_float_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_left_raw_, pub_right_raw_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_left_rect_, pub_right_rect_;
    rclcpp::Publisher<stereo_msgs::msg::DisparityImage>::SharedPtr pub_disparity_;
    rclcpp::Publisher<stereo_vision_msgs::msg::PerformanceMetrics>::SharedPtr pub_performance_;

    rclcpp::TimerBase::SharedPtr timer_, param_timer_;

    int camera_index_;
    std::string calibration_file_;
    bool publish_intermediate_;
    std::string node_namespace_;
    int camera_number_;

    std::thread processing_thread_;
    std::mutex frame_mutex_;
    cv::Mat latest_frame_;
    std::atomic<bool> stop_processing_{false};
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<StereoCameraNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}