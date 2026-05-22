#!/usr/bin/env python3
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from hazmat_vision.hazmat_inference import init_inference, run_frame


class HazmatRosNode(Node):
    def __init__(self):
        super().__init__('hazmat_ros_node')

        self.declare_parameter('camera_index', 0)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('confidence_threshold', 0.4)

        self.declare_parameter('data_path', 'hazmatstuff/Hazmat_Individual')
        self.declare_parameter('weights_path', 'hazmatstuff/hazmat_weights_individual.pth')
        self.declare_parameter('device', 'cpu')  # cpu/cuda/mps

        self.camera_index = int(self.get_parameter('camera_index').value)
        self.fps = float(self.get_parameter('fps').value)
        self.conf_th = float(self.get_parameter('confidence_threshold').value)

        data_path = self.get_parameter('data_path').value
        weights_path = self.get_parameter('weights_path').value
        device = self.get_parameter('device').value

        # Load model once
        init_inference(data_path=data_path, weights_path=weights_path, device_str=device)
        self.get_logger().info("Inference initialized.")

        # ROS publishers
        self.bridge = CvBridge(),
        self.annot_pub = self.create_publisher(Image, '/hazmat/annotated', 10)
        self.labels_pub = self.create_publisher(String, '/hazmat/labels', 10)

        # Camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.camera_index}")

        period = 1.0 / max(self.fps, 1.0)
        self.timer = self.create_timer(period, self.tick)

    def tick(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to read frame")
            return

        annotated, labels = run_frame(frame, confidence_threshold=self.conf_th)

        img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        self.annot_pub.publish(img_msg)

        self.labels_pub.publish(String(data=",".join(labels) if labels else ""))

    def destroy_node(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HazmatRosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()