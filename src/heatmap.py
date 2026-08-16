"""Spatial movement density and heatmap generator based on tracking centroids."""
import cv2
import numpy as np

class HeatmapGenerator:
    def __init__(self, width: int, height: int, decay: float = 0.98):
        self.width = width
        self.height = height
        self.decay = decay
        self.density = np.zeros((height, width), dtype=np.float32)

    def update(self, centroids: list[tuple[int, int]], radius: int = 25):
        self.density *= self.decay
        for cx, cy in centroids:
            if 0 <= cx < self.width and 0 <= cy < self.height:
                cv2.circle(self.density, (cx, cy), radius, 1.0, -1)

    def get_overlay(self, base_frame: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        norm = cv2.normalize(self.density, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color_map = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        return cv2.addWeighted(base_frame, 1.0 - alpha, color_map, alpha, 0)
