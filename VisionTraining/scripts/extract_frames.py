import cv2
from pathlib import Path
import re

VIDEO = "../videos/VIDEO_FILE.mp4" #Configure this for video files
LABEL_DIR = Path("../data/labels/validation")
IMAGE_DIR = Path("../data/images/validation")

IMAGE_DIR.mkdir(exist_ok=True)

# Set this to 0 if CVAT's frame_000000 corresponds to
# the first frame of the video.
FRAME_OFFSET = 0

# Get all label files
label_files = list(LABEL_DIR.glob("*.txt"))

# Map video frame number -> output filename
wanted_frames = {}

for label_file in label_files:
    match = re.search(r"(\d+)$", label_file.stem)

    if not match:
        print(f"Could not determine frame number from {label_file.name}")
        continue

    frame_number = int(match.group(1))
    video_frame = frame_number + FRAME_OFFSET

    wanted_frames[video_frame] = label_file.stem

print(f"Found {len(wanted_frames)} annotated frames.")

# Open video
cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO}")

frame_index = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_index in wanted_frames:
        name = wanted_frames[frame_index]
        output_path = IMAGE_DIR / f"{name}.jpg"

        cv2.imwrite(str(output_path), frame)
        print(f"Saved {output_path}")

    frame_index += 1

cap.release()

print("Done.")