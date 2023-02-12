#include <chrono>
#include <functional>
#include <memory>

#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;

/* This example creates a subclass of Node and uses std::bind() to register a
 * member function as a callback from the timer. */

class MinimalPublisher : public rclcpp::Node {
 public:
  MinimalPublisher() : Node("minimal_publisher"), count_(0) {
    this->declare_parameter("v", 2.0);
    this->declare_parameter("d", 4.0);
    publisher_ =
        this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
            "drive", 1);
    timer_ = this->create_wall_timer(
        10ms, std::bind(&MinimalPublisher::timer_callback, this));
  }

 private:
  void timer_callback() {
    auto message = ackermann_msgs::msg::AckermannDriveStamped();
    float v = this->get_parameter("v").get_parameter_value().get<float>();
    float d = this->get_parameter("d").get_parameter_value().get<float>();
    message.drive.speed = v;
    message.drive.steering_angle = d;
    RCLCPP_INFO(this->get_logger(), "Publishing: '%f'", message.drive.speed);
    publisher_->publish(message);
  }
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr
      publisher_;
  size_t count_;
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalPublisher>());
  rclcpp::shutdown();
  return 0;
}