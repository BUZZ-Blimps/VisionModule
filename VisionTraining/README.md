# Vision training

This directory contains the YOLO training dataset and the scripts used to
extract frames, prepare data, train, evaluate, and run inference for the
balloon detector.

All commands below are run from:

```text
VisionModule\VisionTraining
```

## 1. Install the training dependencies

Use Python 3.10 or newer. A **virtual environment** keeps the training packages
separate from the rest of the project.

```powershell
python -m pip install --upgrade pip
python -m pip install ultralytics opencv-python moviepy
```

An NVIDIA GPU with a working CUDA-enabled PyTorch installation is recommended,
but CPU training also works by using `device=cpu`.
Device is customized in:

```text
scripts/train.py
```

## 2. Required file layout

Create the files and directories below. The `.jpg` images and YOLO `.txt`
labels must have the same stem:

```text
VisionTraining\
├── data\
│   ├── images\
│   │   ├── train\
│   │   │   ├── frame_000000.jpg
│   │   │   └── ...
│   │   └── validation\
│   │       ├── frame_000000.jpg
│   │       └── ...
│   └── labels\
│       ├── train\
│       │   ├── frame_000000.txt
│       │   └── ...
│       └── validation\
│           ├── frame_000000.txt
│           └── ...
├── model\
│   ├── data.yaml
│   ├── train.txt
│   ├── val.txt
│   └── yolov8n.pt
├── scripts\
│   ├── extract_frames.py
│   ├── splitter.py
│   └── train.py
└── videos\
    └── input videos (*.mp4)
```

CVAT was used to create the original dataset:
[https://www.cvat.ai](https://www.cvat.ai)
The exported .zip file from CVAT will include the .yaml and .txt files. These need to be renamed and moved to the correct locations for this to work.

The `.gitignore` ignores raw videos, images, labels, and training
runs. Keep those files locally.

### YOLO label format

Each label file contains one object per line:

```text
class_id center_x center_y width height
```

Coordinates are normalized to `0..1`. The classes configured in this project
are:

```yaml
0: Balloon
1: P Balloon
```

For example, `data/images/train/frame_000123.jpg` must have its annotations
in `data/labels/train/frame_000123.txt`.

## 3. Prepare frames and labels

### Split a video into left and right halves (optional)

`splitter.py` reads the input path near the top of the file and writes two
half-width videos. Configure `INPUT_PATH`, `OUTPUT_PATH_L`, and
`OUTPUT_PATH_R` before running it.

The script's default input is `..\videos\WIN_UTD_Pro.mp4`.

### Extract frames for annotated labels

`scripts\extract_frames.py` extracts only the video frames whose label files
already exist. Configure
`VIDEO`, `LABEL_DIR`, and `IMAGE_DIR` at the top of that script, then run it
from the `scripts` directory.

The default script configuration expects labels under
`data\labels\validation`.

Set `FRAME_OFFSET` if the annotation tool's frame numbering does not start at
the first video frame.

## 4. Configure the dataset

The checked-in `model\train.txt` and `model\val.txt` contain paths beginning
with `data/images/...`, so `model\data.yaml` must use the parent directory as
the dataset root and must reference the list files explicitly. Set its full
contents to:

```yaml
path: ..
train: model/train.txt
val: model/val.txt

names:
  0: Balloon
  1: P Balloon
```

The list files should contain one image path per line, relative to the
`VisionTraining` directory:

```text
data/images/train/frame_000000.jpg
data/images/train/frame_000001.jpg
```

`model\val.txt` should use `data/images/validation/...` paths. Every image
listed must have a matching label in the corresponding `data\labels` folder.

## 5. Train the model

The current `scripts\train.py` loads a previous checkpoint and runs
inference by default; its training block is commented out. The most direct
way to start a new training run is to run the file:

```text
/scripts/train.py
```

Use `device=cpu` if no CUDA GPU is available. Reduce `batch` if the GPU runs
out of memory. `yolov8n.pt` is the small pretrained YOLOv8 checkpoint already
stored in `model`; replace it with another compatible `.pt` checkpoint if
needed.

The important training parameters are:

| Parameter | Purpose |
| --- | --- |
| `data` | Dataset YAML file |
| `epochs` | Maximum number of training passes |
| `imgsz` | Image size used by the model |
| `batch` | Images processed at once |
| `project` / `name` | Output directory and run name |
| `device` | GPU index (`0`) or `cpu` |
| `patience` | Early-stopping wait period |

Training outputs are written to:

```text
runs\detect\models\<run-name>\
├── weights\best.pt
├── weights\last.pt
├── args.yaml
└── results.csv
```

Use `weights\best.pt` for evaluation and inference.

## 6. Evaluate a trained model

Uncomment `model.val()`

Validation plots are saved under `runs\detect\val*`.

## 7. Run inference on a video

Uncomment `model.predict()`

Predictions are saved under a new directory in `runs\detect\predict*`.
`show=True` opens a preview window; use `show=False` on headless machines.

## Common problems

- **Images are not found:** run from `VisionTraining` and verify that
  `model\data.yaml` is exactly the configuration shown above.
- **No labels are found:** verify that every image has a same-named `.txt`
  file in the matching `data\labels\train` or `data\labels\validation`
  directory.
- **CUDA error:** use `device=cpu`, or install a PyTorch build compatible
  with the installed NVIDIA driver and CUDA runtime.
- **The training script performs inference instead:** use the direct training
  command above, or uncomment and update the training block in
  `scripts\train.py` so its `data` value is `model/data.yaml`.