#include "rclcpp/rclcpp.hpp"
#include "stereo_camera_node.hpp"

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StereoCameraNode>());
    rclcpp::shutdown();
    return 0;
}