import rclpy
from rclpy.node import Node
from stereo_msgs.msg import DisparityImage
from stereo_vision_msgs.msg import DetectionArray, Detection
import cv2
import numpy as np
from cv_bridge import CvBridge

class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')

        # Define the namespace for the topics
        self.namespace = 'BurnCreamBlimp'
        self.declare_parameter('disparity_topic', f'/{self.namespace}/disparity')
        self.declare_parameter('detection_topic', f'/{self.namespace}/detections')

        self.bridge = CvBridge()
        
        self.disparity_subscriber = self.create_subscription(
            DisparityImage,
            self.get_parameter('disparity_topic').value,
            self.disparity_callback,
            10
        )
        
        self.detection_subscriber = self.create_subscription(
            DetectionArray,
            self.get_parameter('detection_topic').value,
            self.detection_callback,
            10
        )

        self.current_disparity = None
        self.detections = None

    def disparity_callback(self, msg):
        # Convert DisparityImage to OpenCV format
        self.current_disparity = self.bridge.imgmsg_to_cv2(msg.image, desired_encoding='32FC1')

    def detection_callback(self, msg):
        self.detections = msg

        if self.current_disparity is not None:
            self.visualize_detections()

    def visualize_detections(self):
        # Ensure the image is in the right format for visualization
        disparity_image = (self.current_disparity * 255 / np.max(self.current_disparity)).astype(np.uint8)

        # Convert to color image
        disparity_color = cv2.applyColorMap(disparity_image, cv2.COLORMAP_JET)
        disparity_color = cv2.imread("/home/opi/GitHub/VisionModule/27999c38-frame02366.jpg")
        # Draw detections
        for detection in self.detections.detections:
            bbox = detection.bbox
            label = detection.label
            confidence = detection.confidence
            
            # Convert bbox to integers
            x, y, width, height = map(int, bbox)
            cv2.rectangle(disparity_color, (x, y), (x + width, y + height), (0, 255, 0), 2)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(disparity_color, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Show the image
        cv2.imshow('Disparity Image with Detections', disparity_color)
        cv2.waitKey(1)  # Refresh window

def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
