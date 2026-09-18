import numpy as np
import cv2
from insightface.app import FaceAnalysis


class FaceEmbedder:
    def __init__(self, model_name: str = "buffalo_l", det_size: tuple = (640, 640)):
        self.app = FaceAnalysis(
            name=model_name,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=det_size)

    def get_embedding(self, face_crop: np.ndarray) -> np.ndarray | None:
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        faces = self.app.get(face_rgb)

        if not faces:
            return None

        best = max(faces, key=lambda f: f.det_score)
        return best.embedding

    def get_embeddings(self, face_crops: list[np.ndarray]) -> list[np.ndarray]:
        embeddings = []
        for crop in face_crops:
            emb = self.get_embedding(crop)
            if emb is not None:
                embeddings.append(emb)
        return embeddings
