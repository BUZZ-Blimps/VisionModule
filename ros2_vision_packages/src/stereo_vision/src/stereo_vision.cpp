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
#include "stereo_vision_msgs/msg/detection.hpp"
#include "stereo_vision_msgs/msg/detection_array.hpp"
#include <future>
#include <vector>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>
#include <fstream>
#include <numeric>

struct InputDetails {
    std::vector<std::string> input_names;
    std::vector<int64_t> input_shape;
    int input_height;
    int input_width;
};


#ifdef ENABLE_RKNN
#include <rknn_api.h>
#include "postprocess.h"
#include "preprocess.h"
#include "RgaUtils.h"

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM 10
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25
#define LABEL_NUMB_MAX_SIZE 80

static char* labels[OBJ_CLASS_NUM];

const std::string rknn_model_path_ = ament_index_cpp::get_package_share_directory("stereo_vision") + "/models/yolov5.rknn";

#else

#include <onnxruntime_cxx_api.h>

#define OBJ_CLASS_NUM 10
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25

const std::string onnx_model_path_ = ament_index_cpp::get_package_share_directory("stereo_vision") + "/models/yolov5.onnx";

#endif

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
        updateDisparityMapParams();

        // Use default QoS for image topics
        const auto qos = rclcpp::QoS(10);  // Keep last 10 messages

        // Create publishers with default QoS
        if (publish_intermediate_) {
            pub_left_raw_ = this->create_publisher<sensor_msgs::msg::Image>(node_namespace_ + "/left_raw", qos);
            pub_right_raw_ = this->create_publisher<sensor_msgs::msg::Image>(node_namespace_ + "/right_raw", qos);
            pub_left_rect_ = this->create_publisher<sensor_msgs::msg::Image>(node_namespace_ + "/left_rect", qos);
            pub_right_rect_ = this->create_publisher<sensor_msgs::msg::Image>(node_namespace_ + "/right_rect", qos);
            pub_left_debay_ = this->create_publisher<sensor_msgs::msg::Image>(node_namespace_ + "/left_debay", qos);
            pub_right_debay_ = this->create_publisher<sensor_msgs::msg::Image>(node_namespace_ + "/right_debay", qos);
        }

        // Use default QoS for disparity publisher
        pub_disparity_ = this->create_publisher<stereo_msgs::msg::DisparityImage>(node_namespace_ + "/disparity", qos);

        // Create performance publisher
        pub_performance_ = this->create_publisher<stereo_vision_msgs::msg::PerformanceMetrics>(
            node_namespace_ + "/performance", 10);

        // Create detections publisher
        pub_detections_ = this->create_publisher<stereo_vision_msgs::msg::DetectionArray>(node_namespace_ + "/detections", 10);

        // Create timer for main loop
        timer_ = this->create_wall_timer(std::chrono::milliseconds(33), std::bind(&StereoCameraNode::timerCallback, this));

        // Create timer for updating disparity map parameters
        param_timer_ = this->create_wall_timer(std::chrono::seconds(1), std::bind(&StereoCameraNode::updateDisparityMapParams, this));

        #ifdef ENABLE_RKNN
        // Initialize RKNN
        if (!initRKNN()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize RKNN");
            return;
        }
        #else
        if (!initONNX()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize ONNX Runtime");
            return;
        }
        #endif

        // Initialize the processing thread
        processing_thread_ = std::thread(&StereoCameraNode::processingLoop, this);
    }

    ~StereoCameraNode()
    {
        stop_processing_ = true;
        if (processing_thread_.joinable()) {
            processing_thread_.join();
        }
        #ifdef ENABLE_RKNN
        rknn_destroy(rknn_ctx_);
        #endif
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
        while (rclcpp::ok() && !stop_processing_)
        {
            cv::Mat frame;
            {
                std::lock_guard<std::mutex> lock(frame_mutex_);
                if (latest_frame_.empty()) {
                    continue;
                }
                frame = latest_frame_.clone();
            }

            auto start_time = std::chrono::high_resolution_clock::now();

            auto split_start = std::chrono::high_resolution_clock::now();

            cv::Rect left_roi(0, 0, frame.cols / 2, frame.rows);
            cv::Rect right_roi(frame.cols / 2, 0, frame.cols / 2, frame.rows);
            left_raw_ = frame(left_roi);
            right_raw_ = frame(right_roi);
            auto split_end = std::chrono::high_resolution_clock::now();

            auto rectify_start = std::chrono::high_resolution_clock::now();
            cv::remap(left_raw_, left_rect_, map1_left_, map2_left_, cv::INTER_LINEAR);
            cv::remap(right_raw_, right_rect_, map1_right_, map2_right_, cv::INTER_LINEAR);
            auto rectify_end = std::chrono::high_resolution_clock::now();

            // Start YOLO inference asynchronously
           yolo_future_ = std::async(std::launch::async, &StereoCameraNode::performYOLOInference, this, left_rect_);

            // Compute disparity map
            auto disparity_start = std::chrono::high_resolution_clock::now();

            auto debay_start = std::chrono::high_resolution_clock::now();
            cv::cvtColor(left_rect_, left_debay_, cv::COLOR_BGR2GRAY);
            cv::cvtColor(right_rect_, right_debay_, cv::COLOR_BGR2GRAY);
            auto debay_end = std::chrono::high_resolution_clock::now();
            stereo_->compute(left_debay_, right_debay_, disparity_);
            auto disparity_end = std::chrono::high_resolution_clock::now();

            // Wait for YOLO inference to complete
            auto yolo_start = std::chrono::high_resolution_clock::now();
            auto detections = yolo_future_.get();
            auto yolo_end = std::chrono::high_resolution_clock::now();

            auto end_time = std::chrono::high_resolution_clock::now();

            // Publish results
            publishImages(left_raw_, right_raw_, left_rect_, right_rect_, left_debay_, right_debay_, disparity_);
            publishDetections(detections);
            publishPerformanceMetrics(split_start, split_end, debay_start, debay_end, rectify_start, rectify_end, disparity_start, disparity_end, yolo_start, yolo_end, start_time, end_time);
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

    void publishImages(const cv::Mat& left_raw, const cv::Mat& right_raw, const cv::Mat& left_rect, const cv::Mat& right_rect, const cv::Mat& left_rect_debay, const cv::Mat& right_rect_debay, const cv::Mat& disparity)
    {
        static auto last_publish_time = std::chrono::steady_clock::now();
        auto current_time = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_publish_time).count() >= 100) {
            if (publish_intermediate_) {
                publishImage(left_raw, pub_left_raw_, sensor_msgs::image_encodings::MONO8);
                publishImage(right_raw, pub_right_raw_, sensor_msgs::image_encodings::MONO8);
                publishImage(left_rect, pub_left_rect_, sensor_msgs::image_encodings::BGR8);
                publishImage(right_rect, pub_right_rect_, sensor_msgs::image_encodings::BGR8);
                publishImage(left_rect_debay, pub_left_debay_, sensor_msgs::image_encodings::BGR8);
                publishImage(right_rect_debay, pub_right_debay_, sensor_msgs::image_encodings::BGR8);
            }
            publishDisparityImage(disparity);
            last_publish_time = current_time;
        }
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
        const std::chrono::high_resolution_clock::time_point& yolo_start,
        const std::chrono::high_resolution_clock::time_point& yolo_end,
        const std::chrono::high_resolution_clock::time_point& start_time,
        const std::chrono::high_resolution_clock::time_point& end_time)
    {
        auto msg = stereo_vision_msgs::msg::PerformanceMetrics();

        msg.split_time = std::chrono::duration_cast<std::chrono::microseconds>(split_end - split_start).count() / 1000.0;
        msg.debay_time = std::chrono::duration_cast<std::chrono::microseconds>(debay_end - debay_start).count() / 1000.0;
        msg.rectify_time = std::chrono::duration_cast<std::chrono::microseconds>(rectify_end - rectify_start).count() / 1000.0;
        msg.disparity_time = std::chrono::duration_cast<std::chrono::microseconds>(disparity_end - disparity_start).count() / 1000.0;
        msg.yolo_time = std::chrono::duration_cast<std::chrono::microseconds>(yolo_end - yolo_start).count() / 1000.0;
        msg.total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;

        msg.fps = 1000.0 / msg.total_time;  // Convert ms to fps

        pub_performance_->publish(msg);
    }

    void publishDetections(const std::vector<stereo_vision_msgs::msg::Detection>& detections)
    {
        auto msg = stereo_vision_msgs::msg::DetectionArray();
        msg.header.stamp = this->now();
        msg.header.frame_id = "stereo_camera";
        msg.detections = detections;
        pub_detections_->publish(msg);
    }

    #ifdef ENABLE_RKNN
    bool initRKNN()
    {
        int ret = rknn_init(&rknn_ctx_, const_cast<void*>(static_cast<const void*>(rknn_model_path_.c_str())), 0, 0, NULL);
        if (ret < 0) {
            RCLCPP_ERROR(this->get_logger(), "rknn_init fail! ret=%d", ret);
            return false;
        }

        ret = rknn_query(rknn_ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
        if (ret < 0) {
            RCLCPP_ERROR(this->get_logger(), "rknn_query fail! ret=%d", ret);
            return false;
        }

        rknn_tensor_attr input_attrs[io_num_.n_input];
        memset(input_attrs, 0, sizeof(input_attrs));
        for (int i = 0; i < io_num_.n_input; i++) {
            input_attrs[i].index = i;
            ret = rknn_query(rknn_ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
            if (ret < 0) {
                RCLCPP_ERROR(this->get_logger(), "rknn_query fail! ret=%d", ret);
                return false;
            }
        }

        output_attrs_.resize(io_num_.n_output);
        memset(output_attrs_.data(), 0, sizeof(rknn_tensor_attr) * io_num_.n_output);
        for (int i = 0; i < io_num_.n_output; i++) {
            output_attrs_[i].index = i;
            ret = rknn_query(rknn_ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_[i]), sizeof(rknn_tensor_attr));
            if (ret < 0) {
                RCLCPP_ERROR(this->get_logger(), "rknn_query fail! ret=%d", ret);
                return false;
            }
        }

        model_width_ = input_attrs[0].dims[2];
        model_height_ = input_attrs[0].dims[1];

        scale_w_ = (float)model_width_ / image_size_.width;
        scale_h_ = (float)model_height_ / image_size_.height;

        return true;
    }

    std::vector<stereo_vision_msgs::msg::Detection> performYOLOInference(const cv::Mat& image)
    {
        std::vector<stereo_vision_msgs::msg::Detection> detections;

        // Preprocess image
        cv::Mat resized_image;
        cv::resize(image, resized_image, cv::Size(model_width_, model_height_));
        cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);

        // Set input
        rknn_input inputs[1];
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].size = model_width_ * model_height_ * 3;
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        inputs[0].buf = resized_image.data;

        rknn_inputs_set(rknn_ctx_, 1, inputs);

        // Run inference
        rknn_output outputs[io_num_.n_output];
        memset(outputs, 0, sizeof(outputs));
        for (int i = 0; i < io_num_.n_output; i++)
        {
            outputs[i].index = i;
            outputs[i].want_float = 0;
        }

        int ret = rknn_run(rknn_ctx_, NULL);
        ret = rknn_outputs_get(rknn_ctx_, io_num_.n_output, outputs, NULL);

        // Post-process
        detect_result_group_t detect_result_group;
        std::vector<float> out_scales;
        std::vector<int32_t> out_zps;
        for (int i = 0; i < io_num_.n_output; ++i)
        {
            out_scales.push_back(output_attrs_[i].scale);
            out_zps.push_back(output_attrs_[i].zp);
        }

        float conf_threshold = BOX_THRESH;
        float nms_threshold = NMS_THRESH;
        BOX_RECT scale_w_rect = {0, 0, static_cast<int>(scale_w_), 0};
        ret = post_process((int8_t*)outputs[0].buf, (int8_t*)outputs[1].buf, (int8_t*)outputs[2].buf, 
                               model_height_, model_width_, 
                               conf_threshold, nms_threshold,
                               scale_w_rect, scale_w_, scale_h_, out_zps, out_scales, &detect_result_group);

        // Convert detect_result_group to detections
        for (int i = 0; i < detect_result_group.count; i++)
        {
            stereo_vision_msgs::msg::Detection det;
            det.label = std::string(detect_result_group.results[i].name);
            det.confidence = detect_result_group.results[i].prop;
            det.bbox = {detect_result_group.results[i].box.left, 
                         detect_result_group.results[i].box.top,
                         detect_result_group.results[i].box.right - detect_result_group.results[i].box.left,
                         detect_result_group.results[i].box.bottom - detect_result_group.results[i].box.top};
            detections.push_back(det);
        }

        rknn_outputs_release(rknn_ctx_, io_num_.n_output, outputs);

        return detections;
    }

    #else
    bool initONNX()
    {
        try {
            ort_env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "stereo_vision");
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            ort_session_ = std::make_unique<Ort::Session>(ort_env_, onnx_model_path_.c_str(), session_options);

            Ort::AllocatorWithDefaultOptions allocator;

            size_t num_input_nodes = ort_session_->GetInputCount();
            size_t num_output_nodes = ort_session_->GetOutputCount();

            input_node_names_str_.clear();
            output_node_names_str_.clear();
            input_node_names_.clear();
            output_node_names_.clear();

            for (size_t i = 0; i < num_input_nodes; i++) {
                Ort::AllocatedStringPtr input_name = ort_session_->GetInputNameAllocated(i, allocator);
                input_node_names_str_.push_back(input_name.get());
            }

            for (size_t i = 0; i < num_output_nodes; i++) {
                Ort::AllocatedStringPtr output_name = ort_session_->GetOutputNameAllocated(i, allocator);
                output_node_names_str_.push_back(output_name.get());
            }

            for (const auto& name : input_node_names_str_) {
                input_node_names_.push_back(name.c_str());
            }

            for (const auto& name : output_node_names_str_) {
                output_node_names_.push_back(name.c_str());
            }

            Ort::TypeInfo type_info = ort_session_->GetInputTypeInfo(0);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto input_shape = tensor_info.GetShape();
            model_height_ = input_shape[2];
            model_width_ = input_shape[3];

            scale_w_ = static_cast<float>(model_width_) / image_size_.width;
            scale_h_ = static_cast<float>(model_height_) / image_size_.height;

            return true;
        } catch (const Ort::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "ONNX Runtime error: %s", e.what());
            return false;
        }
    }

    void exportTensorToFile(const std::vector<float>& tensor, const std::vector<int64_t>& shape, const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        
        // Write shape
        int32_t shape_size = static_cast<int32_t>(shape.size());
        file.write(reinterpret_cast<const char*>(&shape_size), sizeof(int32_t));
        for (const auto& dim : shape) {
            int32_t dim32 = static_cast<int32_t>(dim);
            file.write(reinterpret_cast<const char*>(&dim32), sizeof(int32_t));
        }
        
        // Write data
        file.write(reinterpret_cast<const char*>(tensor.data()), tensor.size() * sizeof(float));
        
        file.close();
    }

    std::vector<stereo_vision_msgs::msg::Detection> performYOLOInference(const cv::Mat& image)
    {
        try {
            cv::Mat resized_image;
            cv::resize(image, resized_image, cv::Size(model_width_, model_height_));

            // Convert BGR to RGB and normalize
            cv::Mat rgb_image;
            cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGR2RGB);
            rgb_image.convertTo(rgb_image, CV_32F, 1.0f / 255.0f);

            // Create input tensor values
            std::vector<float> input_tensor_values(1 * 3 * model_height_ * model_width_);
            for (int c = 0; c < 3; ++c) {
                for (int h = 0; h < model_height_; ++h) {
                    for (int w = 0; w < model_width_; ++w) {
                        int index = c * model_height_ * model_width_ + h * model_width_ + w;
                        input_tensor_values[index] = rgb_image.at<cv::Vec3f>(h, w)[c];
                    }
                }
            }

            // Create Ort::MemoryInfo and Ort::Value
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<int64_t> input_shape = {1, 3, static_cast<int64_t>(model_height_), static_cast<int64_t>(model_width_)};
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                input_tensor_values.data(),
                input_tensor_values.size(),
                input_shape.data(),
                input_shape.size()
            );

            // Run ONNX inference
            std::vector<Ort::Value> output_tensors;
            {
                std::lock_guard<std::mutex> lock(ort_mutex_);
                output_tensors = ort_session_->Run(
                    Ort::RunOptions{nullptr},
                    input_node_names_.data(),
                    &input_tensor,
                    1,
                    output_node_names_.data(),
                    output_node_names_.size()
                );
            }

            // Print output tensor shape
            Ort::TypeInfo type_info = output_tensors[0].GetTypeInfo();
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> output_dims = tensor_info.GetShape();
            
            size_t output_size = std::accumulate(output_dims.begin(), output_dims.end(), 1, std::multiplies<int64_t>());
            Eigen::Map<Eigen::MatrixXf> output_matrix(output_tensors[0].GetTensorMutableData<float>(), output_dims[2], output_dims[1]);
            auto detections = process_output(output_matrix, *ort_session_, conf_threshold_, iou_threshold_);
            return detections;

        } catch (const Ort::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "ONNX Runtime inference error: %s", e.what());
            return std::vector<stereo_vision_msgs::msg::Detection>();
        }
    }
    #endif

    InputDetails get_input_details(Ort::Session& session) {
        InputDetails details;
        Ort::AllocatorWithDefaultOptions allocator;

        size_t num_input_nodes = session.GetInputCount();
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session.GetInputNameAllocated(i, allocator);
            details.input_names.push_back(input_name.get());
        }

        auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        details.input_shape = input_shape;
        details.input_height = input_shape[2];
        details.input_width = input_shape[3];

        return details;
    }

    std::vector<int> nms(const Eigen::MatrixXf& boxes, const Eigen::VectorXf& scores, float iou_threshold) {
        std::vector<int> sorted_indices(scores.size());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(),
                [&scores](int i1, int i2) { return scores(i1) > scores(i2); });

        std::vector<int> keep_boxes;
        while (!sorted_indices.empty()) {
            int box_id = sorted_indices[0];
            keep_boxes.push_back(box_id);

            Eigen::VectorXf ious = compute_iou(boxes.row(box_id), boxes);

            std::vector<int> keep_indices;
            for (size_t i = 1; i < sorted_indices.size(); ++i) {
                if (ious(sorted_indices[i]) < iou_threshold) {
                    keep_indices.push_back(sorted_indices[i]);
                }
            }
            sorted_indices = keep_indices;
        }
        return keep_boxes;
    }

    std::vector<int> multiclass_nms(const Eigen::MatrixXf& boxes, const Eigen::VectorXf& scores, 
                                    const Eigen::VectorXi& class_ids, float iou_threshold) {
        std::vector<int> unique_class_ids;
        for (int i = 0; i < class_ids.size(); ++i) {
            if (std::find(unique_class_ids.begin(), unique_class_ids.end(), class_ids(i)) == unique_class_ids.end()) {
                unique_class_ids.push_back(class_ids(i));
            }
        }

        std::vector<int> keep_boxes;

        for (int class_id : unique_class_ids) {
            std::vector<int> class_indices;
            for (int i = 0; i < class_ids.size(); ++i) {
                if (class_ids(i) == class_id) {
                    class_indices.push_back(i);
                }
            }

            Eigen::MatrixXf class_boxes(class_indices.size(), 4);
            Eigen::VectorXf class_scores(class_indices.size());
            for (size_t i = 0; i < class_indices.size(); ++i) {
                class_boxes.row(i) = boxes.row(class_indices[i]);
                class_scores(i) = scores(class_indices[i]);
            }

            std::vector<int> class_keep_boxes = nms(class_boxes, class_scores, iou_threshold);
            for (int idx : class_keep_boxes) {
                keep_boxes.push_back(class_indices[idx]);
            }
        }

        return keep_boxes;
    }

    Eigen::VectorXf compute_iou(const Eigen::VectorXf& box, const Eigen::MatrixXf& boxes) {
        Eigen::VectorXf xmin = boxes.col(0).cwiseMax(box(0));
        Eigen::VectorXf ymin = boxes.col(1).cwiseMax(box(1));
        Eigen::VectorXf xmax = boxes.col(2).cwiseMin(box(2));
        Eigen::VectorXf ymax = boxes.col(3).cwiseMin(box(3));

        Eigen::VectorXf intersection_area = (xmax - xmin).cwiseMax(0.0f).cwiseProduct(ymax - ymin).cwiseMax(0.0f);

        float box_area = (box(2) - box(0)) * (box(3) - box(1));
        Eigen::VectorXf boxes_area = (boxes.col(2) - boxes.col(0)).cwiseProduct(boxes.col(3) - boxes.col(1));
        
        // Create a vector of the same size as boxes_area, filled with box_area
        Eigen::VectorXf box_area_vector = Eigen::VectorXf::Constant(boxes_area.size(), box_area);
        
        Eigen::VectorXf union_area = boxes_area + box_area_vector - intersection_area;

        return intersection_area.cwiseQuotient(union_area);
    }

    Eigen::MatrixXf xywh2xyxy(const Eigen::MatrixXf& x) {
        Eigen::MatrixXf y = x;
        y.col(0) = x.col(0) - x.col(2) / 2.0f;
        y.col(1) = x.col(1) - x.col(3) / 2.0f;
        y.col(2) = x.col(0) + x.col(2) / 2.0f;
        y.col(3) = x.col(1) + x.col(3) / 2.0f;
        return y;
    }

    Eigen::MatrixXf extract_boxes(const Eigen::MatrixXf& predictions, int input_width, int input_height, 
                                int img_width, int img_height) {
        Eigen::MatrixXf boxes = predictions.leftCols(4);
        boxes = rescale_boxes(boxes, input_width, input_height, img_width, img_height);
        return xywh2xyxy(boxes);
    }

    Eigen::MatrixXf rescale_boxes(const Eigen::MatrixXf& boxes, int input_width, int input_height, 
                                int img_width, int img_height) {
        Eigen::Vector4f input_shape(input_width, input_height, input_width, input_height);
        Eigen::Vector4f img_shape(img_width, img_height, img_width, img_height);
        return (boxes.array().rowwise() / input_shape.transpose().array()).rowwise() * img_shape.transpose().array();
    }

    std::vector<stereo_vision_msgs::msg::Detection> process_output(const Eigen::MatrixXf& output, Ort::Session& session, 
                                                                  float conf_threshold, float iou_threshold) {
        std::vector<stereo_vision_msgs::msg::Detection> detections;

        InputDetails input_details = get_input_details(session);
        Eigen::MatrixXf predictions = output;

        // Calculate scores (maximum of class probabilities)
        Eigen::VectorXf scores = predictions.rightCols(predictions.cols() - 4).rowwise().maxCoeff();

        // Create a vector to store indices of predictions above the confidence threshold
        std::vector<int> keep_indices;
        for (int i = 0; i < scores.size(); ++i) {
            if (scores(i) > conf_threshold) {
                keep_indices.push_back(i);
            }
        }

        // Create new matrices for filtered predictions and scores
        Eigen::MatrixXf filtered_predictions(keep_indices.size(), predictions.cols());
        Eigen::VectorXf filtered_scores(keep_indices.size());

        for (size_t i = 0; i < keep_indices.size(); ++i) {
            filtered_predictions.row(i) = predictions.row(keep_indices[i]);
            filtered_scores(i) = scores(keep_indices[i]);
        }

        predictions = filtered_predictions;
        scores = filtered_scores;

        // Calculate class IDs for the filtered predictions
        Eigen::VectorXi class_ids = predictions.rightCols(predictions.cols() - 4).rowwise().maxCoeff().cast<int>();

        // Extract boxes from the filtered predictions
        Eigen::MatrixXf boxes = extract_boxes(predictions, input_details.input_width, input_details.input_height, 
                                            img_width_, img_height_);

        // Apply NMS (Non-Maximum Suppression)
        std::vector<int> nms_indices = multiclass_nms(boxes, scores, class_ids, iou_threshold);

        //Log the number of detections
        RCLCPP_INFO(this->get_logger(), "Number of detections: %zu", nms_indices.size());

        for (int idx : nms_indices) {
            stereo_vision_msgs::msg::Detection detection;
            detection.label = std::to_string(class_ids(idx));
                detection.bbox[0] = boxes(idx, 0);
                detection.bbox[1] = boxes(idx, 1);
                detection.bbox[2] = boxes(idx, 2);
                detection.bbox[3] = boxes(idx, 3);
                detection.confidence = scores(idx);
                detections.push_back(detection);
            }

        return detections;
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
    cv::Mat left_rect_gray_, right_rect_gray_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_left_raw_, pub_right_raw_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_left_rect_, pub_right_rect_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_left_debay_, pub_right_debay_;
    rclcpp::Publisher<stereo_msgs::msg::DisparityImage>::SharedPtr pub_disparity_;
    rclcpp::Publisher<stereo_vision_msgs::msg::PerformanceMetrics>::SharedPtr pub_performance_;
    rclcpp::Publisher<stereo_vision_msgs::msg::DetectionArray>::SharedPtr pub_detections_;

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

    #ifdef ENABLE_RKNN
        rknn_context rknn_ctx_;
        rknn_input_output_num io_num_;
        std::vector<rknn_tensor_attr> output_attrs_;
    #else
        Ort::Env ort_env_{ORT_LOGGING_LEVEL_WARNING, "stereo_vision"};
        std::unique_ptr<Ort::Session> ort_session_;
        std::vector<const char*> input_node_names_;
        std::vector<const char*> output_node_names_;
        std::vector<std::string> input_node_names_str_;
        std::vector<std::string> output_node_names_str_;
        std::mutex ort_mutex_;
    #endif
    
    int model_width_;
    int model_height_;
    float scale_w_;
    float scale_h_;

    std::future<std::vector<stereo_vision_msgs::msg::Detection>> yolo_future_;

    float conf_threshold_ = 0.25f;  // You can adjust this value
    float iou_threshold_ = 0.9f;   // You can adjust this value
    int img_width_ = 1280;  // Set this to your image width
    int img_height_ = 960;  // Set this to your image height
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<StereoCameraNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}