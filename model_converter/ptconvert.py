from ultralytics import YOLO
# Load your trained model
model = YOLO ("../weights/best.pt")
# Export to RKNN format , targeting the RK3588 chip
model.export( format = "rknn", name = "rk3588")

ros2 launch stereo_vision stereo_vision.launch.py \
model_path:=/root/best_model/best_v11-rk3588.rknn