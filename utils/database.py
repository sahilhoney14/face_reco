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
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    self.data = json.loads(content) if content else {}
            except Exception as e:
                print(f"[WARN] Failed to load database '{self.db_path}': {e}. Initializing empty database.")
                self.data = {}

    def _save(self):
        parent_dir = os.path.dirname(os.path.abspath(self.db_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

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
