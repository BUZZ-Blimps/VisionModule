// stereo_camera_node.cpp

#include "stereo_camera_node.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include "publishers.hpp"
#include "yolo_inference.hpp"
#include <opencv2/imgproc.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

StereoCameraNode::StereoCameraNode() : Node("stereo_camera_node")
{
    // Declare parameters
    this->declare_parameter("camera_index", 0);
    this->declare_parameter("calibration_file", "");
    this->declare_parameter("publish_intermediate", true);
    this->declare_parameter("save_frames", false);
    this->declare_parameter("node_namespace", "BurnCreamBlimp");
    this->declare_parameter("camera_number", 4);
    this->declare_parameter("model_path", "yolov10n_v2.rknn");

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
    frame_save_ = this->get_parameter("save_frames").as_bool();
    node_namespace_ = this->get_parameter("node_namespace").as_string();
    camera_number_ = this->get_parameter("camera_number").as_int();
    model_path_ = this->get_parameter("model_path").as_string();

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
    left_debay_ = cv::Mat(height, width/2, CV_8UC3);
    right_debay_ = cv::Mat(height, width/2, CV_8UC3);
    left_rect_ = cv::Mat(height, width/2, CV_8UC3);
    right_rect_ = cv::Mat(height, width/2, CV_8UC3);
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
    updateDisparityMapParams(stereo_, this);

    std::cout << "Disparity maps generated." << std::endl;

    // Use default QoS for image topics
    const auto qos = rclcpp::QoS(rclcpp::KeepLast(2))
    .reliability(rclcpp::ReliabilityPolicy::Reliable)
    .durability(rclcpp::DurabilityPolicy::Volatile)
    .history(rclcpp::HistoryPolicy::KeepLast)
    .deadline(std::chrono::milliseconds(100))
    .lifespan(rclcpp::Duration(1, 0))  // 1 second
    .liveliness(rclcpp::LivelinessPolicy::Automatic)
    .avoid_ros_namespace_conventions(false);

    // Create publishers with default QoS
    if (publish_intermediate_) {
        pub_left_rect_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(node_namespace_ + "/left_rect/compressed", qos);
        pub_right_rect_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(node_namespace_ + "/right_rect/compressed", qos);
    }

    RCLCPP_INFO(this->get_logger(), "Value of frame saver: %d", frame_save_);
    if (frame_save_) {
        saver.initVideoSaver(15, true);
        
    }

    // Use default QoS for disparity publisher
    pub_disparity_ = this->create_publisher<stereo_msgs::msg::DisparityImage>(node_namespace_ + "/disparity", qos);

    // Create performance publisher
    pub_performance_ = this->create_publisher<stereo_vision_msgs::msg::PerformanceMetrics>(
        node_namespace_ + "/performance", 10);

    // Create detections publisher
    pub_detections_ = this->create_publisher<stereo_vision_msgs::msg::DetectionArray>(node_namespace_ + "/detections", 10);
    pub_targets_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(node_namespace_ + "/targets", 10);

    // Create timer for main loop
    timer_ = this->create_wall_timer(std::chrono::milliseconds(33), std::bind(&StereoCameraNode::timerCallback, this));

    // Create timer for updating disparity map parameters
    param_timer_ = this->create_wall_timer(std::chrono::seconds(1), [this]() {updateDisparityMapParams(this->stereo_, this);});


    if (!initRKNN(model_path_, rknn_app_ctx, this->get_logger())) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize RKNN");
        return;
    }

    // Initialize the processing thread
    processing_thread_ = std::thread(&StereoCameraNode::processingLoop, this);

    // In the constructor
    clahe_ = cv::createCLAHE(2.0, cv::Size(8, 8));
}

StereoCameraNode::~StereoCameraNode()
{
    stop_processing_ = true;
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
    rknn_destroy(rknn_app_ctx.rknn_ctx);
}

void StereoCameraNode::timerCallback()
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

void StereoCameraNode::processingLoop() {
    RCLCPP_INFO(this->get_logger(), "Starting processing loop");
    while (rclcpp::ok() && !stop_processing_) {
        try {
            cv::Mat frame;
            {
                std::lock_guard<std::mutex> lock(frame_mutex_);
                if (latest_frame_.empty()) {
                    RCLCPP_DEBUG(this->get_logger(), "No new frame available, continuing to next iteration");
                    continue;
                }
                frame = latest_frame_.clone();
            }

            RCLCPP_DEBUG(this->get_logger(), "Processing new frame");
            auto start_time = std::chrono::high_resolution_clock::now();
            auto split_start = std::chrono::high_resolution_clock::now();

            cv::Rect left_roi(0, 0, frame.cols / 2, frame.rows);
            cv::Rect right_roi(frame.cols / 2, 0, frame.cols / 2, frame.rows);
            left_raw_ = frame(left_roi);
            right_raw_ = frame(right_roi);

            auto split_end = std::chrono::high_resolution_clock::now();
            RCLCPP_DEBUG(this->get_logger(), "Frame split completed");

            cv::Mat resized_image, padded_image;
            cv::resize(left_rect_, resized_image, cv::Size(640, 480), cv::INTER_LINEAR);

            int top = 0;
            int bottom = 640 - 480;
            int left = 0;
            int right = 0;

            cv::copyMakeBorder(resized_image, padded_image, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
            RCLCPP_DEBUG(this->get_logger(), "Starting YOLO inference asynchronously");
            yolo_future_ = std::async(std::launch::async, performYOLOInference, std::ref(padded_image), std::ref(rknn_app_ctx), box_conf_threshold, nms_threshold, this->get_logger());

            auto rectify_start = std::chrono::high_resolution_clock::now();

            RCLCPP_DEBUG(this->get_logger(), "Starting image rectification");
            cv::remap(left_raw_, left_rect_, map1_left_, map2_left_, cv::INTER_NEAREST);
            cv::resize(left_rect_, left_rect_, cv::Size(640,480), cv::INTER_LINEAR);
            cv::cvtColor(left_rect_, left_debay_, cv::COLOR_BGR2GRAY);

            cv::remap(right_raw_, right_rect_, map1_right_, map2_right_, cv::INTER_NEAREST);
            cv::resize(right_rect_, right_rect_, cv::Size(640,480), cv::INTER_LINEAR);
            cv::cvtColor(right_rect_, right_debay_, cv::COLOR_BGR2GRAY);

            auto rectify_end = std::chrono::high_resolution_clock::now();
            RCLCPP_DEBUG(this->get_logger(), "Image rectification completed");

            RCLCPP_DEBUG(this->get_logger(), "Applying CLAHE");
            cv::Mat left_clahe, right_clahe;
            clahe_->apply(left_debay_, left_clahe);
            clahe_->apply(right_debay_, right_clahe);

            RCLCPP_DEBUG(this->get_logger(), "Computing disparity map");
            auto disparity_start = std::chrono::high_resolution_clock::now();
            stereo_->compute(left_clahe, right_clahe, disparity_);
            auto disparity_msg = convertToDisparityImageMsg(disparity_, this);
            auto disparity_end = std::chrono::high_resolution_clock::now();

            RCLCPP_DEBUG(this->get_logger(), "Waiting for YOLO inference to complete");
            auto yolo_start = std::chrono::high_resolution_clock::now();
            auto detections = yolo_future_.get();
            auto yolo_end = std::chrono::high_resolution_clock::now();

            RCLCPP_DEBUG(this->get_logger(), "Estimating depth");
            estimateDepth(detections, *disparity_msg);

            auto end_time = std::chrono::high_resolution_clock::now();

            cv::resize(left_rect_, left_rect_, cv::Size(320,240), cv::INTER_LINEAR);
            cv::resize(right_rect_, right_rect_, cv::Size(320,240), cv::INTER_LINEAR);

            RCLCPP_DEBUG(this->get_logger(), "Publishing results");
            publishImages(this, left_rect_, right_rect_, disparity_msg);
            publishDetections(this, detections);
            publishPerformanceMetrics(this, split_start, split_end, rectify_start, rectify_end, disparity_start, disparity_end, yolo_start, yolo_end, start_time, end_time);

            if (frame_save_) {
                RCLCPP_DEBUG(this->get_logger(), "Saving frame");
                saver.writeFrame(padded_image);
            }

            RCLCPP_DEBUG(this->get_logger(), "Frame processing completed");
        } catch (const cv::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "OpenCV error occurred: %s", e.what());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error occurred: %s", e.what());
        } catch (...) {
            RCLCPP_ERROR(this->get_logger(), "Unknown error occurred");
        }
    }
    RCLCPP_INFO(this->get_logger(), "Processing loop ended");
}