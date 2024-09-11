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
find_package(Eigen3 REQUIRED)

include_directories(${stereo_vision_msgs_INCLUDE_DIRS})
include_directories(${CMAKE_INSTALL_PREFIX}/include/stereo_vision_msgs)
include_directories(${EIGEN3_INCLUDE_DIR})


# Add RKNPU2 dependencies
set(RKNPU2_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/rknpu2)
set(RGA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/rknpu2/runtime/RGA)

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

include_directories(${stereo_vision_msgs_INCLUDE_DIRS})
include_directories(${CMAKE_INSTALL_PREFIX}/include/stereo_vision_msgs)
include_directories(include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/rknpu2/3rdparty/rga/include)


set(SOURCES src/stereo_vision.cpp src/postprocess.cpp src/rk_common.cpp)
add_executable(stereo_vision ${SOURCES})

ament_target_dependencies(stereo_vision
  rclcpp
  sensor_msgs
  stereo_msgs
  cv_bridge
  OpenCV
  stereo_vision_msgs
)

# Link the postprocess library to the main executable
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
)

target_link_libraries(stereo_vision
  ${CMAKE_CURRENT_SOURCE_DIR}/external/rknpu2/runtime/Linux/librknn_api/librknnrt.so
  ${CMAKE_CURRENT_SOURCE_DIR}/external/rknpu2/3rdparty/rga/librga.so
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

install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/models/yolov10n.rknn
  DESTINATION share/${PROJECT_NAME}/models
)

ament_package()