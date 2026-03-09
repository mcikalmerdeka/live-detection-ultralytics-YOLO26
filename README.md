# Live Object Detection with Ultralytics YOLO26

A browser-based object detection application built with [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) and [Gradio](https://www.gradio.app/). The application supports static image and video inference as well as real-time detection from a webcam feed.

## Features

- **Image detection** — upload an image and receive an annotated result with bounding boxes and class labels.
- **Video detection** — upload a video file and stream the annotated output progressively in 2-second chunks.
- **Live webcam detection** — stream frames from your webcam and see detections rendered in real time.
- **Model selector** — switch between three YOLO26 variants (n / m / x) without restarting the app. Models are cached after the first load.
- **Adjustable thresholds** — confidence and IoU thresholds are configurable per inference run.

## Models

| Variant | File size | Parameters | mAP50-95 | CPU speed    |
| ------- | --------- | ---------- | -------- | ------------ |
| YOLO26n | 5.3 MB    | 2.4M       | 40.9     | 38.9 ms/img  |
| YOLO26m | 42.2 MB   | 20.4M      | 53.1     | 220.0 ms/img |
| YOLO26x | 113 MB    | 55.7M      | 57.5     | 525.8 ms/img |

Models are downloaded automatically from the [Ultralytics GitHub releases](https://github.com/ultralytics/assets/releases/tag/v8.4.0) on first use and stored in the `models/` directory.

## Requirements

- Python 3.12+
- Dependencies listed in `requirements.txt`

```
ultralytics>=8.4.21
gradio>=6.9.0
opencv-python>=4.13.0.92
pillow>=12.1.1
```

## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/mcikalmerdeka/live-detection-ultralytics-YOLO26.git
cd live-detection-ultralytics-YOLO26
```

**2. Install dependencies**

using uv:

```bash
uv sync or uv add -r requirements.txt
```

using pip:

```bash
pip install -r requirements.txt
```

**3. Run the application**

```bash
python app.py
```

The app will be available at `http://localhost:7860` in your browser.

## Project Structure

```
.
├── app.py              # Main application
├── models/             # Downloaded YOLO26 weights (.pt files)
├── requirements.txt
└── pyproject.toml
```

## Notes

- For real-time webcam detection, grant camera access when prompted by the browser.
- On CPU, the YOLO26n variant is recommended for the webcam tab to maintain acceptable frame rates. YOLO26m and YOLO26x are better suited for image and video inference or GPU environments.
- Video output files are written as temporary `.mp4` chunks in the working directory during processing.

## License

This project uses Ultralytics YOLO26 which is licensed under [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
