"""
YOLOv8-based object detector with OpenCV visualization.
Supports image, video file, and live webcam input.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ObjectDetector:
    """
    Wraps YOLOv8 with a clean detect/draw API.

    Example::

        det = ObjectDetector("yolov8n.pt", conf=0.45)
        annotated = det.detect_image("photo.jpg", save_path="out.jpg")
        det.detect_video("clip.mp4", save_path="out.mp4")
        det.detect_webcam()          # press 'q' to quit
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.40,
        iou: float = 0.45,
        device: str = "cpu",
    ) -> None:
        logger.info("Loading YOLO model: %s", model_path)
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.device = device

    def _run(self, source) -> List:
        return self.model.predict(
            source, conf=self.conf, iou=self.iou, device=self.device, verbose=False
        )

    def _annotate(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw bounding boxes, labels, and confidence scores."""
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{self.model.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        return frame

    def detect_image(
        self, image_path: str, save_path: Optional[str] = None
    ) -> np.ndarray:
        img = cv2.imread(image_path)
        results = self._run(img)
        annotated = self._annotate(img.copy(), results)
        if save_path:
            cv2.imwrite(save_path, annotated)
            logger.info("Saved annotated image: %s", save_path)
        return annotated

    def detect_video(self, video_path: str, save_path: Optional[str] = None) -> None:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = None
        if save_path:
            writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self._run(frame)
            annotated = self._annotate(frame, results)
            if writer:
                writer.write(annotated)
        cap.release()
        if writer:
            writer.release()
            logger.info("Saved annotated video: %s", save_path)

    def detect_webcam(self, camera_id: int = 0) -> None:
        cap = cv2.VideoCapture(camera_id)
        logger.info("Webcam stream started. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self._run(frame)
            annotated = self._annotate(frame, results)
            cv2.imshow("YOLOv8 Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
