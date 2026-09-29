import re
from pathlib import Path

import cv2


SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO = SCRIPT_DIR.parent / "videos" / "left_half123.mp4"

# Set this to 0 if CVAT's frame_000000 corresponds to the first video frame.
FRAME_OFFSET = 0


def get_dataset_directories(split: str) -> tuple[Path, Path, Path]:
    """Return the image, label, and model-list paths for a dataset split."""
    if split not in {"train", "val"}:
        raise ValueError("split must be either 'train' or 'val'.")

    dataset_dir = SCRIPT_DIR.parent
    data_split = "validation" if split == "val" else split
    return (
        dataset_dir / "data" / "images" / data_split,
        dataset_dir / "data" / "labels" / data_split,
        dataset_dir / "model" / f"{split}.txt",
    )


def update_model_image_list(model_list: Path, image_paths: list[Path]) -> None:
    """Add image paths to a YOLO model list without creating duplicates."""
    model_list.parent.mkdir(parents=True, exist_ok=True)
    existing_paths = {
        line.strip()
        for line in model_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    } if model_list.exists() else set()

    relative_paths = [
        path.relative_to(SCRIPT_DIR.parent).as_posix() for path in image_paths
    ]
    new_paths = [path for path in relative_paths if path not in existing_paths]
    if new_paths:
        with model_list.open("a", encoding="utf-8") as output:
            if existing_paths:
                output.write("\n")
            output.write("\n".join(new_paths))
            output.write("\n")


def extract_frames_with_txt(
    video_path: str | Path,
    split: str = "train",
    frame_offset: int = 0,
) -> int:
    """Extract the video frames that have matching YOLO label files."""
    image_dir, label_dir, model_list = get_dataset_directories(split)
    image_dir.mkdir(parents=True, exist_ok=True)

    wanted_frames: dict[int, str] = {}
    for label_file in label_dir.glob("*.txt"):
        match = re.search(r"(\d+)$", label_file.stem)
        if match is None:
            print(f"Could not determine frame number from {label_file.name}")
            continue

        frame_number = int(match.group(1))
        wanted_frames[frame_number + frame_offset] = label_file.stem

    print(f"Found {len(wanted_frames)} annotated frames.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    extracted_count = 0
    extracted_images: list[Path] = []
    frame_index = 0
    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            if frame_index in wanted_frames:
                name = wanted_frames[frame_index]
                output_path = image_dir / f"{name}.jpg"
                if not cv2.imwrite(str(output_path), frame):
                    raise RuntimeError(f"Could not write image: {output_path}")
                extracted_count += 1
                extracted_images.append(output_path)
                print(f"Saved {output_path}")

            frame_index += 1
    finally:
        capture.release()

    update_model_image_list(model_list, extracted_images)
    return extracted_count


def extract_frames_no_txt(
    video_path: str | Path,
    split: str = "train",
    target_fps: float = 60.0,
) -> int:
    """Extract frames at target_fps and create a matching empty YOLO label file."""
    if target_fps <= 0:
        raise ValueError("target_fps must be greater than zero.")

    image_dir, label_dir, model_list = get_dataset_directories(split)
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        capture.release()
        raise RuntimeError(f"Could not determine the FPS for video: {video_path}")

    extracted_count = 0
    extracted_images: list[Path] = []
    frame_index = 0
    next_output_frame = 0
    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            # Select the source frame nearest to each target 1/target_fps interval.
            requested_source_frame = round(next_output_frame * source_fps / target_fps)
            if frame_index >= requested_source_frame:
                name = f"empty_frame_{extracted_count:06d}"
                image_path = image_dir / f"{name}.jpg"
                label_path = label_dir / f"{name}.txt"

                if not cv2.imwrite(str(image_path), frame):
                    raise RuntimeError(f"Could not write image: {image_path}")
                label_path.write_text("", encoding="utf-8")

                extracted_count += 1
                extracted_images.append(image_path)
                next_output_frame += 1

            frame_index += 1
    finally:
        capture.release()

    update_model_image_list(model_list, extracted_images)
    print(
        f"Saved {extracted_count} frames to {image_dir} "
        f"and matching labels to {label_dir}; updated {model_list}."
    )
    return extracted_count


if __name__ == "__main__":
    extract_frames_no_txt(VIDEO, split="train")
    #extract_frames_with_txt(VIDEO, split="train", frame_offset=0)