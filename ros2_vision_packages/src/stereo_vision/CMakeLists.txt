cmake_minimum_required(VERSION 3.8)
project(stereo_vision)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(stereo_msgs REQUIRED)  # Add this line
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)
find_package(stereo_vision_msgs REQUIRED)
include_directories(${stereo_vision_msgs_INCLUDE_DIRS})
include_directories(${CMAKE_INSTALL_PREFIX}/include/stereo_vision_msgs)

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

ament_package()