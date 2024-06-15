#ifndef PEDESTRAIN_DETECTOR_NODE_HPP_
#define PEDESTRAIN_DETECTOR_NODE_HPP_

// std
#include <memory>
// rclcpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
// SDL
#include <SDL_mixer.h>
// project
#include "pedestrian_detector/yolov8.hpp"
#include "rm_interfaces/msg/gimbal_cmd.hpp"

namespace pdn {

class PedestrianDetectorNode : public rclcpp::Node {
public:
  PedestrianDetectorNode(const rclcpp::NodeOptions &options);

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;

  void traffic_light_callback(
      const std_srvs::srv::SetBool::Request::SharedPtr request,
      std_srvs::srv::SetBool::Response::SharedPtr response);
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr traffic_light_srv_;
  bool traffic_light_on_;

  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
      camera_info_sub_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<rm_interfaces::msg::GimbalCmd>::SharedPtr gimbal_pub_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  Mix_Music *alert_audio_;
  std::future<void> alert_audio_future_;

  Mix_Music * red_light_audio_;
  Mix_Music * green_light_audio_;

  std::unique_ptr<yolov8::YOLOv8> detector_;
  double fx_, fy_, cx_, cy_;
};

} // namespace pdn
#endif
