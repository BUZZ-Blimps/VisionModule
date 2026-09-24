from moviepy.editor import VideoFileClip
import moviepy.video.fx.all as vfx
from pathlib import Path


INPUT_PATH = "../videos/WIN_20260908_16_22_48_Pro.mp4"
OUTPUT_PATH_L = "left_half67.mp4"
OUTPUT_PATH_R = "right_half67.mp4"

def unused_output_path(output_path):
    path = Path(output_path)
    if not path.exists():
        return str(path)

    counter = 1
    while True:
        candidate = path.with_name(
            f"{path.stem} ({counter}){path.suffix}"
        )
        if not candidate.exists():
            return str(candidate)
        counter += 1

def split_video_halves(input_path, output_left, output_right):
    # Load the original video
    clip = VideoFileClip(input_path)
    width, height = clip.size

    # Calculate the midpoint for a vertical split
    mid_x = width // 2

    # Crop the left half: (x1, y1, x2, y2)
    # Starts at (0,0) and goes to (mid_x, height)
    left_clip = vfx.crop(clip, x1=0, y1=0, x2=mid_x, y2=height)

    # Crop the right half
    # Starts at (mid_x, 0) and goes to (width, height)
    right_clip = vfx.crop(clip, x1=mid_x, y1=0, x2=width, y2=height)

    # Write the results to files
    left_clip.write_videofile(unused_output_path(output_left), codec="libx264")
    right_clip.write_videofile(unused_output_path(output_right), codec="libx264")

    # Close clips to release resources
    clip.close()
    left_clip.close()
    right_clip.close()

# Example usage
split_video_halves(INPUT_PATH, OUTPUT_PATH_L, OUTPUT_PATH_R)