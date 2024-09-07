cmake_minimum_required(VERSION 3.8)
project(stereo_vision)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
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

# Check if we're building for RK3588-based devices
if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64" AND EXISTS "/proc/device-tree/model")
  file(READ "/proc/device-tree/model" DEVICE_MODEL)
  string(STRIP "${DEVICE_MODEL}" DEVICE_MODEL)
  if(DEVICE_MODEL MATCHES "RK3588" OR
     DEVICE_MODEL MATCHES "Orange Pi 5" OR
     DEVICE_MODEL MATCHES "Orange Pi 5B^@" OR
     DEVICE_MODEL MATCHES "Rock 5")
    set(IS_RK3588 TRUE)
    message(STATUS "Detected RK3588-based device: ${DEVICE_MODEL}")
    add_definitions(-DENABLE_RKNN)
  else()
    set(IS_RK3588 FALSE)
    message(STATUS "Not an RK3588-based device. YOLO object detection will be disabled.")
  endif()
else()
  set(IS_RK3588 FALSE)
  message(STATUS "Not an RK3588-based device. YOLO object detection will be disabled.")
endif()

# RKNN-related setup only for RK3588-based devices
if(IS_RK3588)
  # Add RKNPU2 dependencies
  set(RKNPU2_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/rknpu2)
  set(RGA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/rknpu2/runtime/RGA)

  find_package(PkgConfig REQUIRED)
  pkg_check_modules(RKNN REQUIRED rknn_api)

  include_directories(
    ${RKNPU2_DIR}/runtime/RK3588/Linux/librknn_api/include
    ${RGA_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/rknpu2/runtime/RK3588/Linux/librknn_api/include
    ${RKNN_INCLUDE_DIRS}
  )

  link_directories(
    ${RKNPU2_DIR}/runtime/RK3588/Linux/librknn_api/aarch64
    ${RGA_DIR}/libs
    ${CMAKE_CURRENT_SOURCE_DIR}/external/rknpu2/runtime/RK3588/Linux/librknn_api/aarch64
  )

  add_definitions(-DENABLE_RKNN)
else()
  message(STATUS "Not an RK3588-based device. YOLO object detection will be disabled.")
endif()

include_directories(${stereo_vision_msgs_INCLUDE_DIRS})
include_directories(${CMAKE_INSTALL_PREFIX}/include/stereo_vision_msgs)
include_directories(include)

add_executable(stereo_vision src/stereo_vision.cpp)
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
)

if(IS_RK3588)
  target_link_libraries(stereo_vision
    rknn_api
    rga
    ${RKNN_LIBRARIES}
  )
endif()

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

if(IS_RK3588)
  install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/models/yolov5.rknn
    DESTINATION share/${PROJECT_NAME}/models
  )
endif()

ament_package()