import cv2
import numpy as np
from ultralytics import YOLO


class FaceDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_faces(self, frame: np.ndarray) -> list[dict]:
        results = self.model(frame, verbose=False)
        faces = []

        for result in results:
            for box in result.boxes: #type:ignore
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls != 0 or conf < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size == 0:
                    continue

                faces.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "face_crop": face_crop,
                })

        return faces

    def detect_faces_from_image(self, image_path: str) -> list[dict]:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Tidak bisa membaca gambar: {image_path}")
        return self.detect_faces(frame)
