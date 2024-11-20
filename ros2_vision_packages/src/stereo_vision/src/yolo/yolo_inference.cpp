#include "yolo_inference.hpp"
#include "yolo_postprocess.hpp"

bool initRKNN(const std::string& model_path, rknn_app_context_t& app_ctx, rclcpp::Logger logger)
{
    memset(&app_ctx, 0, sizeof(rknn_app_context_t));

    int ret = rknn_init(&app_ctx.rknn_ctx, const_cast<void*>(static_cast<const void*>(model_path.c_str())), 0, 0, NULL);
    if (ret < 0) {
        RCLCPP_ERROR(logger, "rknn_init fail! ret=%d", ret);
        return false;
    }

    ret = rknn_query(app_ctx.rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &app_ctx.io_num, sizeof(app_ctx.io_num));
    if (ret < 0) {
        RCLCPP_ERROR(logger, "rknn_query fail! ret=%d", ret);
        return false;
    }

    app_ctx.input_attrs = (rknn_tensor_attr*)malloc(app_ctx.io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx.output_attrs = (rknn_tensor_attr*)malloc(app_ctx.io_num.n_output * sizeof(rknn_tensor_attr));

    for (uint32_t i = 0; i < app_ctx.io_num.n_input; i++) {
        app_ctx.input_attrs[i].index = i;
        ret = rknn_query(app_ctx.rknn_ctx, RKNN_QUERY_INPUT_ATTR, &(app_ctx.input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            RCLCPP_ERROR(logger, "rknn_query fail! ret=%d", ret);
            return false;
        }
    }

    for (uint32_t i = 0; i < app_ctx.io_num.n_output; i++) {
        app_ctx.output_attrs[i].index = i;
        ret = rknn_query(app_ctx.rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &(app_ctx.output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            RCLCPP_ERROR(logger, "rknn_query fail! ret=%d", ret);
            return false;
        }
    }

    app_ctx.model_width = app_ctx.input_attrs[0].dims[2];
    app_ctx.model_height = app_ctx.input_attrs[0].dims[1];
    app_ctx.model_channel = app_ctx.input_attrs[0].dims[3];

    if (app_ctx.output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && app_ctx.output_attrs[0].type != RKNN_TENSOR_FLOAT16) {
        app_ctx.is_quant = true;
    } else {
        app_ctx.is_quant = false;
    }

    if (app_ctx.input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        RCLCPP_INFO(logger, "model input is NCHW");
        app_ctx.model_channel = app_ctx.input_attrs[0].dims[1];
        app_ctx.model_height = app_ctx.input_attrs[0].dims[2];
        app_ctx.model_width = app_ctx.input_attrs[0].dims[3];
    } else {
        RCLCPP_INFO(logger, "model input is NHWC");
        app_ctx.model_height = app_ctx.input_attrs[0].dims[1];
        app_ctx.model_width = app_ctx.input_attrs[0].dims[2];
        app_ctx.model_channel = app_ctx.input_attrs[0].dims[3];
    }

    RCLCPP_INFO(logger, "model input height=%d, width=%d, channel=%d", 
        app_ctx.model_height, app_ctx.model_width, app_ctx.model_channel);

    return true;
}

std::vector<stereo_vision_msgs::msg::Detection> performYOLOInference(const cv::Mat& image, rknn_app_context_t& app_ctx, float box_conf_threshold, float nms_threshold, rclcpp::Logger logger)
{
    int ret;
    std::vector<stereo_vision_msgs::msg::Detection> detections;
    cv::Mat input_image;
    
    try {
        cv::cvtColor(image, input_image, cv::COLOR_BGR2RGB);

        rknn_input inputs[app_ctx.io_num.n_input];
        rknn_output outputs[app_ctx.io_num.n_output];
        object_detect_result_list od_results;   

        memset(inputs, 0, sizeof(inputs));
        memset(outputs, 0, sizeof(outputs));

        // Set input
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].size = app_ctx.model_width * app_ctx.model_height * app_ctx.model_channel;
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        inputs[0].buf = input_image.data;

        ret = rknn_inputs_set(app_ctx.rknn_ctx, app_ctx.io_num.n_input, inputs);
        if (ret < 0) {
            throw std::runtime_error("rknn_input_set failed");
        }

        ret = rknn_run(app_ctx.rknn_ctx, nullptr);
        if (ret < 0) {
            throw std::runtime_error("rknn_run failed");
        }
        
        for (uint32_t i = 0; i < app_ctx.io_num.n_output; i++) {
            outputs[i].index = i;
            outputs[i].want_float = (!app_ctx.is_quant);
        }

        ret = rknn_outputs_get(app_ctx.rknn_ctx, app_ctx.io_num.n_output, outputs, NULL);
        if (ret < 0) {
            throw std::runtime_error("rknn_outputs_get failed");
        }

        post_process(&app_ctx, outputs, box_conf_threshold, nms_threshold, 1.0, 1.0, &od_results);

        // Define minimum confidence and area thresholds
        const float MIN_CONFIDENCE = 0.6f;  // Adjust this value as needed
        const int MIN_AREA = 100;  // Adjust this value as needed (e.g., 50x20 pixels)

        for (int i = 0; i < od_results.count; i++) {
            object_detect_result* det_result = &(od_results.results[i]);

            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;
            int cls = det_result->cls_id;
            float prob = det_result->prop;

            // Calculate bounding box area
            int area = (x2 - x1) * (y2 - y1);

            // Filter based on confidence and area
            if (prob >= MIN_CONFIDENCE && area >= MIN_AREA && cls < 8) {
                stereo_vision_msgs::msg::Detection detection;
                detection.class_id = cls;
                detection.bbox[0] = x1;
                detection.bbox[1] = y1;
                detection.bbox[2] = x2 - x1;
                detection.bbox[3] = y2 - y1;
                detection.confidence = prob;
                detections.push_back(detection);
            }
        }

        rknn_outputs_release(app_ctx.rknn_ctx, app_ctx.io_num.n_output, outputs);
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger, "Error during YOLO inference: %s", e.what());
    } catch (...) {
        RCLCPP_ERROR(logger, "Unknown error occurred during YOLO inference");
    }

    return detections;
}