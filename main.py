"""Unified CLI runner for Computer Vision tasks: Detection, Tracking, Counting."""
import argparse
from src.detector import ObjectDetector
from src.tracker import ObjectTracker

def main():
    parser = argparse.ArgumentParser(description="Computer Vision Pipeline CLI")
    parser.add_argument("--mode", choices=["detect", "track", "webcam"], default="webcam")
    parser.add_argument("--source", type=str, default="0", help="Path to image/video or webcam index")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.45)
    args = parser.parse_args()

    if args.mode == "webcam":
        det = ObjectDetector(model_path=args.model, conf=args.conf)
        det.detect_webcam(camera_id=int(args.source) if args.source.isdigit() else 0)
    elif args.mode == "detect":
        det = ObjectDetector(model_path=args.model, conf=args.conf)
        det.detect_image(args.source, save_path="output.jpg")
    elif args.mode == "track":
        tracker = ObjectTracker(model_path=args.model, conf=args.conf)
        tracker.track_video(args.source, save_path="output_tracked.mp4")

if __name__ == "__main__":
    main()
