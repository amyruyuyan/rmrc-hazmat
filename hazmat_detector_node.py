import cv2
import numpy as np 
import rclpy
from rclpy.node import Node 

from sensor_msgs.msg import Image
from std_msgs.msg import String 
from cv_bridge import CvBridge 

# also assuming for now that i can import my methods from the original code 
# but modified for ease of import which i'll do later

# from hazmat_vision.hazmat_inference import init_inference, run_frame

from hazmat_vision.hazmat_inference import init_inference, run_frame

# i think there is supposed to be something here and this file is definitely not
# right because i couldn't figure out the package thing

class HazmatCameraNode(Node):
    def __init__(self):
        print("NODE INIT STARTED")

        super().__init__('hazmat_ros_node')
        print("SUPER INIT DONE")

        self.declare_parameter('camera_id', 0)
        self.declare_parameter('confidence_threshold', 0.4)
        print("PARAMETERS DECLARED")

        self.declare_parameter(
            'data_path', 
            '/root/ros2_ws/src/hazmat_vision/hazmat_vision/hazmatstuff/Hazmat_Individual'
        ) 
        self.declare_parameter(
            'weights_path', 
            '/root/ros2_ws/src/hazmat_vision/hazmat_vision/hazmatstuff/hazmat_weights_individual.pth'
        )         
        self.declare_parameter('device', 'cpu')

        self.camera_id = int(self.get_parameter('camera_id').value)
        print(f"CAMERA ID = {self.camera_id}")

        self.conf_th = float(self.get_parameter('confidence_threshold').value)

        data_path = self.get_parameter('data_path').value
        weights_path = self.get_parameter('weights_path').value 
        device = self.get_parameter('device').value

        self.camera_topic = f'/cameras/raw/camera_{self.camera_id}'
        self.annotated_topic = f'/hazmat/annotated/camera_{self.camera_id}'
        self.labels_topic = f'/hazmat/labels/camera_{self.camera_id}'

        self.get_logger().info(f'subscribing to: {self.camera_topic}')
        self.get_logger().info(f'publishing annotated images to: {self.annotated_topic}')
        self.get_logger().info(f'publishing labels to: {self.labels_topic}')

        # load the model
        init_inference(data_path=data_path, weights_path=weights_path, device_str=device)
        print("INFERENCE LOADED")
        self.get_logger().info("inference intialized")

        # ros publishers
        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(Image, self.camera_topic, self.image_callback, 10)
        self.annot_pub = self.create_publisher(Image, self.annotated_topic, 10)
        self.labels_pub = self.create_publisher(String, self.labels_topic, 10)

        self.get_logger().info('hazmat detector node ready')

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"failed to convert ros image to opencv: {e}")
            return

        try:
            annotated_frame, labels = run_frame(
                frame,
                confidence_threshold=self.conf_th
            )
        except Exception as e:
            self.get_logger().error(f"hazmat inference failed: {e}")

        if labels: 
            self.get_logger().info(f"detected: {labels}")

            try:
                annotated_msg = self.bridge.cv2_to_imgmsg(
                    annotated_frame,
                    encoding = 'bgr8'
                )
            except Exception as e:
                self.get_logger().error(f"failed to publish annotated image: {e}")

            self.labels_pub.publish(
                String(data=",".join(labels))
            )

def main(args=None):
    rclpy.init(args=args)
    node = HazmatCameraNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


# removed stuff

'''
# camera
self.cap = cv2.VideoCapture(self.camera_id)
print("VIDEOCAPTURE CREATED")

print(f"CAMERA OPENED? {self.camera_id}")

if not self.cap.isOpened():
    raise RuntimeError(f"cannot open camera {self.camera_id}")

period = 1.0 / max(self.fps, 1.0)
self.timer = self.create_timer(period, self.tick)
'''

'''
def tick(self):
# print("TICK STARTED")

ret, frame = self.cap.read()
# print(f"READ SUCCESS = {ret}")

if not ret:
    self.get_logger().warn("failed to read frame")
    return 

annotated, labels = run_frame(frame, confidence_threshold=self.conf_th)
if labels:
    print(f"DETECTED LABELS = {labels}")

# still need to add annot_pub and labels_pub stuff here

img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
self.annot_pub.publish(img_msg)
# print("ANNOTATED IMAGE PUBLISHED")

self.labels_pub.publish(String(data=",".join(labels) if labels else ""))
# print("LABELS PUBLISHED")


def destroy_node(self):
super().destroy_node()
'''
