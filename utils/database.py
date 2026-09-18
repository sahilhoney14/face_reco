import json
import os
import numpy as np


class FaceDatabase:
    def __init__(self, db_path: str = "face_database.json"):
        self.db_path = db_path
        self.data: dict[str, list[list[float]]] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "r") as f:
                self.data = json.load(f)

    def _save(self):
        with open(self.db_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def register(self, name: str, embedding: np.ndarray):
        emb_list = embedding.tolist()
        if name in self.data:
            self.data[name].append(emb_list)
        else:
            self.data[name] = [emb_list]
        self._save()

    def get_all_embeddings(self) -> dict[str, list[np.ndarray]]:
        result = {}
        for name, embs in self.data.items():
            result[name] = [np.array(e) for e in embs]
        return result

    def get_names(self) -> list[str]:
        return list(self.data.keys())

    def remove(self, name: str) -> bool:
        if name in self.data:
            del self.data[name]
            self._save()
            return True
        return False

    def count(self) -> int:
        return len(self.data)
