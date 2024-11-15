import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import argparse
from stereo_vision_msgs.msg import DetectionArray, Detection  # Replace 'your_package' with the actual package name

class ImageSubscriber(Node):
    def __init__(self, namespace):
        super().__init__('image_subscriber')
        self.namespace = namespace
        self.cv_bridge = CvBridge()
        self.latest_image = None
        self.latest_detections = None

        # Subscribe to the compressed image topic
        self.image_subscription = self.create_subscription(
            CompressedImage,
            f'/{namespace}/left_rect/compressed',
            self.image_callback,
            10)

        # Subscribe to the detections topic
        self.detections_subscription = self.create_subscription(
            DetectionArray,
            f'/{namespace}/detections',
            self.detections_callback,
            10)

    def image_callback(self, msg):
        self.latest_image = cv2.resize(self.cv_bridge.compressed_imgmsg_to_cv2(msg), (0,0), fx=2.0, fy=2.0)
        self.process_and_display()

    def detections_callback(self, msg):
        self.latest_detections = msg.detections
        self.process_and_display()

    def process_and_display(self):
        if self.latest_image is None or self.latest_detections is None:
            return

        image_with_detections = self.latest_image.copy()

        for detection in self.latest_detections:
            x, y, w, h = detection.bbox
            x, y, w, h = int(x), int(y), int(w), int(h)

            # Draw bounding box
            cv2.rectangle(image_with_detections, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Prepare text with detection information
            text = f"Class: {detection.class_id}, Depth: {detection.depth:.2f}m"
            text += f"\nConf: {detection.confidence:.2f}, ID: {detection.track_id}"

            # Draw text background
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image_with_detections, (x, y - text_height - 4), (x + text_width, y), (0, 255, 0), -1)

            # Draw text
            y_offset = y - text_height - 2
            for line in text.split('\n'):
                cv2.putText(image_with_detections, line, (x, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                y_offset += text_height + 2

        cv2.imshow("Detections", image_with_detections)
        cv2.waitKey(1)

def main():
    parser = argparse.ArgumentParser(description='ROS2 Image Subscriber with Detections')
    parser.add_argument('namespace', type=str, help='Namespace for the topics')
    args = parser.parse_args()

    rclpy.init()
    image_subscriber = ImageSubscriber(args.namespace)
    
    try:
        rclpy.spin(image_subscriber)
    except KeyboardInterrupt:
        pass
    
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
