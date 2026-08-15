"""Multi-object tracker using SORT algorithm on top of YOLO detections."""
from __future__ import annotations
import logging
import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ObjectTracker:
    """
    Persistent object tracking across video frames using YOLOv8 + ByteTrack.

    Example::

        tracker = ObjectTracker("yolov8n.pt")
        tracker.track_video("input.mp4", save_path="tracked.mp4")
    """

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.4) -> None:
        self.model = YOLO(model_path)
        self.conf = conf
        self._colors: dict = {}

    def _color(self, track_id: int) -> tuple:
        if track_id not in self._colors:
            np.random.seed(track_id)
            self._colors[track_id] = tuple(np.random.randint(50, 255, 3).tolist())
        return self._colors[track_id]

    def track_video(self, video_path: str, save_path: str | None = None) -> None:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) if save_path else None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model.track(frame, conf=self.conf, persist=True, verbose=False)
            if results[0].boxes.id is not None:
                for box, tid in zip(results[0].boxes.xyxy, results[0].boxes.id.int()):
                    x1, y1, x2, y2 = map(int, box)
                    color = self._color(int(tid))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID {int(tid)}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if writer:
                writer.write(frame)

        cap.release()
        if writer:
            writer.release()
