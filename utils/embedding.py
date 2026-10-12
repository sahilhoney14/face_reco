import numpy as np
import cv2
import onnxruntime as ort
from insightface.app import FaceAnalysis


class FaceEmbedder:
    def __init__(self, model_name: str = "buffalo_l", det_size: tuple = (640, 640)):
        # Dynamically select execution provider based on system availability
        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
        else:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1

        self.app = FaceAnalysis(
            name=model_name,
            providers=providers,
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def get_embedding(self, face_crop: np.ndarray) -> np.ndarray | None:
        if face_crop is None or face_crop.size == 0:
            return None

        try:
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            faces = self.app.get(face_rgb)

            # Fallback: if no face is detected, add padding around crop borders.
            # SCRFD detector can miss faces that touch the crop perimeter.
            if not faces:
                h, w = face_rgb.shape[:2]
                pad_h = max(10, int(h * 0.2))
                pad_w = max(10, int(w * 0.2))
                padded_rgb = cv2.copyMakeBorder(
                    face_rgb, pad_h, pad_h, pad_w, pad_w,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
                faces = self.app.get(padded_rgb)

            if not faces:
                return None

            best = max(faces, key=lambda f: f.det_score)
            return best.embedding
        except Exception as e:
            print(f"[WARN] Error extracting face embedding: {e}")
            return None

    def get_embeddings(self, face_crops: list[np.ndarray]) -> list[np.ndarray]:
        embeddings = []
        for crop in face_crops:
            emb = self.get_embedding(crop)
            if emb is not None:
                embeddings.append(emb)
        return embeddings
