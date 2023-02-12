#include <chrono>
#include <functional>
#include <memory>

#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using std::placeholders::_1;
using namespace std::chrono_literals;

class MinimalSubscriber : public rclcpp::Node {
 public:
  MinimalSubscriber() : Node("minimal_subscriber") {
    subscription_ =
        this->create_subscription<ackermann_msgs::msg::AckermannDriveStamped>(
            "/drive", 1,
            std::bind(&MinimalSubscriber::topic_callback, this, _1));
    publisher_ =
        this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
            "drive_relay", 1);
  }

 private:
  void topic_callback(
      const ackermann_msgs::msg::AckermannDriveStamped::SharedPtr msg) const {
    ackermann_msgs::msg::AckermannDriveStamped driver_msg_relay_;
    driver_msg_relay_.drive.speed = 3 * (msg->drive.speed);
    driver_msg_relay_.drive.steering_angle = 3 * (msg->drive.steering_angle);
    RCLCPP_INFO(this->get_logger(), "Relayed: '%f'\n",
                driver_msg_relay_.drive.speed);
    publisher_->publish(driver_msg_relay_);
  }

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr
      publisher_;
  rclcpp::Subscription<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr
      subscription_;
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalSubscriber>());
  rclcpp::shutdown();
  return 0;
}