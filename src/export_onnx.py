"""Export YOLOv8 PyTorch weights to ONNX Runtime and TensorRT format for edge deployment."""
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

def export_onnx(model_path: str = "yolov8n.pt", imgsz: int = 640, half: bool = True):
    logger.info("Loading PyTorch model for ONNX conversion: %s", model_path)
    model = YOLO(model_path)
    exported_file = model.export(
        format="onnx",
        imgsz=imgsz,
        half=half,
        dynamic=True,
        simplify=True
    )
    logger.info("ONNX model exported to: %s", exported_file)
    return exported_file
