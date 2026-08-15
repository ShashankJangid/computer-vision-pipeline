# computer-vision-pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B00?style=flat-square)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

Real-time object detection and multi-object tracking pipeline built on **YOLOv8** (Ultralytics) and **OpenCV**. Supports static images, video files, and live webcam streams with persistent object identity across frames.

## Features

- **ObjectDetector** — YOLOv8 inference with bounding box + label overlay
- **ObjectTracker** — ByteTrack-based persistent tracking with per-ID colors
- Supports: image, video file, live webcam
- Easily swap any YOLOv8 variant (`yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`)

## Quick Start

```bash
pip install -r requirements.txt

# Detect in a single image
python -c "
from src.detector import ObjectDetector
det = ObjectDetector('yolov8n.pt', conf=0.45)
det.detect_image('photo.jpg', save_path='output.jpg')
"

# Track objects in a video
python -c "
from src.tracker import ObjectTracker
tracker = ObjectTracker('yolov8n.pt')
tracker.track_video('input.mp4', save_path='tracked.mp4')
"

# Live webcam
python -c "
from src.detector import ObjectDetector
ObjectDetector('yolov8n.pt').detect_webcam()
"
```

## License

MIT
