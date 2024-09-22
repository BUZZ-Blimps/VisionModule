import rclpy
from rclpy.node import Node
from stereo_msgs.msg import DisparityImage
from stereo_vision_msgs.msg import DetectionArray
from std_msgs.msg import Float32MultiArray
import numpy as np

class DisparityToDepthNode(Node):
	def __init__(self):
		super().__init__('disparity_to_depth_node')
		self.subscription = self.create_subscription(
		DisparityImage,
		'/BurnCreamBlimp/disparity',
		self.disparity_callback,
		10)
		self.detection_subscription = self.create_subscription(
		DetectionArray,
		'/BurnCreamBlimp/detections',  # Adjust this topic name as needed
		self.detection_callback,
		10)
		self.depth_publisher = self.create_publisher(Float32MultiArray, 'detection_depths', 10)
		self.latest_detections = None
		self.latest_depth = None

	def detection_callback(self, msg):
		self.latest_detections = msg.detections
		if self.latest_depth is not None:
			self.process_detections()

	def disparity_callback(self, msg):
		f = msg.f
		T = msg.t
		disparity = np.frombuffer(msg.image.data, dtype=np.float32).reshape(
		msg.image.height, msg.image.width)
		depth = np.zeros_like(disparity)
		mask = disparity > msg.min_disparity
		depth[mask] = f * T / disparity[mask]
		self.latest_depth = depth
		if self.latest_detections is not None:
			self.process_detections()

	def process_detections(self):
		depth_values = []
		for detection in self.latest_detections:
			bbox = detection.bbox
			x, y, width, height = bbox[0], bbox[1], bbox[2], bbox[3]

			# Calculate the center of the bounding box
			center_x, center_y = int(x + width/2), int(y + height/2)

			# Define a smaller region around the center (e.g., 20% of the bounding box size)
			region_width = max(int(width * 0.2), 1)
			region_height = max(int(height * 0.2), 1)

			x1 = max(0, center_x - region_width // 2)
			y1 = max(0, center_y - region_height // 2)
			x2 = min(self.latest_depth.shape[1], center_x + region_width // 2)
			y2 = min(self.latest_depth.shape[0], center_y + region_height // 2)

			# Extract depth values from the central region
			center_depth = self.latest_depth[y1:y2, x1:x2]

			# Filter out invalid depth values
			valid_depths = center_depth[(center_depth > 0) & (center_depth < np.inf)]

			if valid_depths.size > 0:
				# Use the 10th percentile instead of median to favor closer depths
				depth = float(np.percentile(valid_depths, 10))
				depth_values.append(depth)
			else:
				depth_values.append(float('nan'))

		depth_msg = Float32MultiArray()
		depth_msg.data = depth_values
		self.depth_publisher.publish(depth_msg)

def main(args=None):
	rclpy.init(args=args)
	node = DisparityToDepthNode()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
