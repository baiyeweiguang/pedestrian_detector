#include "pedestrian_detector/pedestrian_detector_node.hpp"

#include <filesystem>

#include <SDL.h>
#include <SDL_mixer.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/qos.hpp>

#define SAMPLE_RATE 44100
#define FRAMES_PER_BUFFER 256

namespace pdn {
PedestrianDetectorNode::PedestrianDetectorNode(
    const rclcpp::NodeOptions &options)
    : Node("pedestrian_detector_node", options) {

  if (SDL_Init(SDL_INIT_AUDIO) < 0) {
    RCLCPP_ERROR(get_logger(), "SDL_Init failed: %s", SDL_GetError());
    return;
  }

  if (Mix_OpenAudio(44100, MIX_DEFAULT_FORMAT, 2, 2048) < 0) {
    RCLCPP_ERROR(get_logger(), "Mix_OpenAudio failed: %s", Mix_GetError());
    return;
  }

  std::string audio_name =
      declare_parameter<std::string>("alert_audio", "audio.mp3");
  std::string audio_path =
      std::filesystem::path(
          ament_index_cpp::get_package_share_directory("pedestrian_detector")) /
      "audio" / audio_name;
  alert_audio_ = Mix_LoadMUS(audio_path.c_str());

  audio_name =
      declare_parameter<std::string>("red_light_audio", "red_light.mp3");
  audio_path =
      std::filesystem::path(
          ament_index_cpp::get_package_share_directory("pedestrian_detector")) /
      "audio" / audio_name;
  red_light_audio_ = Mix_LoadMUS(audio_path.c_str());

  audio_name =
      declare_parameter<std::string>("green_light_audio", "green_light.mp3");
  audio_path =
      std::filesystem::path(
          ament_index_cpp::get_package_share_directory("pedestrian_detector")) /
      "audio" / audio_name;
  green_light_audio_ = Mix_LoadMUS(audio_path.c_str());

  if (alert_audio_ == nullptr || red_light_audio_ == nullptr ||
      green_light_audio_ == nullptr) {
    RCLCPP_ERROR(get_logger(), "Mix_LoadMUS failed: %s", Mix_GetError());
    return;
  }

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  std::string model_name =
      declare_parameter<std::string>("model_name", "yolov8.onnx");
  std::string model_path =
      std::filesystem::path(
          ament_index_cpp::get_package_share_directory("pedestrian_detector")) /
      "model" / model_name;

  std::vector<std::string> labels =
      declare_parameter<std::vector<std::string>>("labels", {"person"});

  detector_ = std::make_unique<yolov8::YOLOv8>(model_path, labels, "CPU");

  camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      "camera_info", rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        fx_ = msg->k[0];
        fy_ = msg->k[4];
        cx_ = msg->k[2];
        cy_ = msg->k[5];
        camera_info_sub_.reset();
      });

  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      "image_raw", rclcpp::SensorDataQoS(),
      std::bind(&PedestrianDetectorNode::image_callback, this,
                std::placeholders::_1));

  traffic_light_srv_ = create_service<std_srvs::srv::SetBool>(
      "traffic_light",
      std::bind(&PedestrianDetectorNode::traffic_light_callback, this,
                std::placeholders::_1, std::placeholders::_2));
  traffic_light_on_ = false;

  gimbal_pub_ = create_publisher<rm_interfaces::msg::GimbalCmd>(
      "cmd_gimbal", rclcpp::SensorDataQoS());
  image_pub_ = create_publisher<sensor_msgs::msg::Image>(
      "debug_img", rclcpp::SensorDataQoS());
}

void PedestrianDetectorNode::image_callback(
    const sensor_msgs::msg::Image::SharedPtr msg) {
  if (camera_info_sub_ != nullptr) {
    RCLCPP_WARN(get_logger(), "No camera info received yet.");
    return;
  }

  double current_yaw, current_pitch;
  try {
    auto t =
        tf_buffer_->lookupTransform("odom", "gimbal_link", msg->header.stamp);
    tf2::Matrix3x3 m(
        tf2::Quaternion(t.transform.rotation.x, t.transform.rotation.y,
                        t.transform.rotation.z, t.transform.rotation.w));
    double roll;
    m.getRPY(roll, current_pitch, current_yaw);
    current_pitch = -current_pitch;
  } catch (tf2::TransformException &ex) {
    RCLCPP_WARN(get_logger(), "%s", ex.what());
    return;
  }

  cv::Mat image = cv_bridge::toCvCopy(msg, "bgr8")->image;
  auto objects = detector_->detect(image);

  objects.erase(std::remove_if(objects.begin(), objects.end(),
                               [](const yolov8::Object &object) {
                                 return object.class_name != "person";
                               }),
                objects.end());

  for (const auto &object : objects) {
    cv::rectangle(image, object.bbox, cv::Scalar(0, 255, 0), 2);
    cv::circle(image,
               cv::Point(object.bbox.x + object.bbox.width * 0.5,
                         object.bbox.y + object.bbox.height * 0.5),
               8, cv::Scalar(0, 0, 255), 2);
    cv::putText(image, object.class_name,
                cv::Point(object.bbox.x, object.bbox.y),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
  }
  cv::circle(image, cv::Point(cx_, cy_), 12, cv::Scalar(255, 255, 255), 2);

  std::sort(objects.begin(), objects.end(),
            [](const yolov8::Object &a, const yolov8::Object &b) {
              return a.bbox.area() > b.bbox.area();
            });

  if (!objects.empty() && traffic_light_on_) {
    auto &final_object = objects.at(0);
    rm_interfaces::msg::GimbalCmd gimbal_cmd;

    double x = final_object.bbox.x + final_object.bbox.width * 0.5;
    double y = final_object.bbox.y + final_object.bbox.height * 0.5;

    using std::chrono_literals::operator""s;
    if (!alert_audio_future_.valid() ||
        alert_audio_future_.wait_for(0s) == std::future_status::ready) {
      alert_audio_future_ = std::async(std::launch::async, [this]() {
        Mix_PlayMusic(alert_audio_, 0);

        while (Mix_PlayingMusic()) {
          SDL_Delay(100);
        }
      });
    }

    gimbal_cmd.pitch = -atan((y - cy_) / fy_) + current_pitch;
    gimbal_cmd.yaw = -atan((x - cx_) / fx_) + current_yaw;
    gimbal_cmd.distance = 1.0;
    gimbal_pub_->publish(gimbal_cmd);
  } else {
    Mix_HaltMusic();
  }

  image_pub_->publish(
      *cv_bridge::CvImage(msg->header, "bgr8", image).toImageMsg());
}

void PedestrianDetectorNode::traffic_light_callback(
    const std_srvs::srv::SetBool::Request::SharedPtr request,
    std_srvs::srv::SetBool::Response::SharedPtr response) {
  traffic_light_on_ = request->data;
  response->success = true;

  if (traffic_light_on_) {
    Mix_PlayMusic(red_light_audio_, 0);
    while (Mix_PlayingMusic()) {
      SDL_Delay(100);
    }
  } else {
    Mix_PlayMusic(green_light_audio_, 0);
    while (Mix_PlayingMusic()) {
      SDL_Delay(100);
    }
  }
}

} // namespace pdn

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(pdn::PedestrianDetectorNode)