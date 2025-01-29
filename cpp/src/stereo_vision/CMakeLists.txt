cmake_minimum_required(VERSION 3.8)
project(stereo_vision)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-write-strings -Wno-unused-parameter -Wno-unused-function -Wno-vla -Wno-sign-compare)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(stereo_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)
find_package(stereo_vision_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

# Add RKNPU2 dependencies
set(RKNPU2_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/rknpu2)

include_directories(
  include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include/utils
  ${CMAKE_CURRENT_SOURCE_DIR}/include/yolo
  ${stereo_vision_msgs_INCLUDE_DIRS}
  ${CMAKE_INSTALL_PREFIX}/include/stereo_vision_msgs
  ${RKNPU2_DIR}/runtime/Linux/librknn_api/include
  ${CMAKE_CURRENT_SOURCE_DIR}/external/rknpu2/runtime/Linux/librknn_api/include/
  ${RKNN_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}

)

link_directories(
  ${RKNPU2_DIR}/runtime/Linux/librknn_api
  ${CMAKE_CURRENT_SOURCE_DIR}/external/rknpu2/runtime/Linux/librknn_api
)

set(SOURCES
  src/main.cpp  
  src/stereo_camera_node.cpp
  src/utils/publishers.cpp
  src/utils/image_processing.cpp
  src/yolo/yolo_inference.cpp
  src/yolo/yolo_postprocess.cpp
  src/yolo/yolo_common.cpp
)

add_executable(stereo_vision ${SOURCES})

ament_target_dependencies(stereo_vision
  rclcpp
  sensor_msgs
  stereo_msgs
  cv_bridge
  OpenCV
  stereo_vision_msgs
)

target_link_libraries(stereo_vision
  ${rclcpp_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${cv_bridge_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${ament_index_cpp_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  ${Threads_LIBRARIES}
  ${stereo_msgs_LIBRARIES}
  ${stereo_vision_msgs_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  OpenMP::OpenMP_CXX
  ${CMAKE_CURRENT_SOURCE_DIR}/external/rknpu2/runtime/Linux/librknn_api/librknnrt.so
)

set_target_properties(stereo_vision PROPERTIES
  INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/external/rknpu2/runtime/Linux/librknn_api"
)

install(TARGETS
  stereo_vision
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  calibration
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY
  models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()