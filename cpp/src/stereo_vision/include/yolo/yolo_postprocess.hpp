#ifndef _RKNN_POSTPROCESS_H_
#define _RKNN_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include "rknn_api.h"
#include "yolo_common.hpp"

#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM 10
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25
#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)

int post_process(rknn_app_context_t *app_ctx, void *outputs, float conf_threshold, float nms_threshold, float scale_w, float scale_h, object_detect_result_list *od_results);

#endif