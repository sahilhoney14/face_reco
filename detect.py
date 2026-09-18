import argparse
import sys

import cv2
import numpy as np

from utils import FaceDetector, FaceEmbedder, FaceDatabase


COLOR_KNOWN = (0, 255, 0)
COLOR_UNKNOWN = (0, 0, 255)
THRESHOLD = 0.6


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def recognize_face(embedding: np.ndarray, db: FaceDatabase,
                   threshold: float = THRESHOLD) -> tuple[str, float]:
    best_name = "Unknown"
    best_score = 0.0

    for name, embeddings in db.get_all_embeddings().items():
        for db_emb in embeddings:
            score = cosine_similarity(embedding, db_emb)
            if score > best_score:
                best_score = score
                best_name = name

    if best_score >= threshold:
        return best_name, best_score
    return "Unknown", best_score


def process_frame(frame: np.ndarray, detector: FaceDetector,
                  embedder: FaceEmbedder, db: FaceDatabase,
                  threshold: float) -> np.ndarray:
    faces = detector.detect_faces(frame)
    results = []

    for face in faces:
        emb = embedder.get_embedding(face["face_crop"])
        if emb is None:
            continue

        name, score = recognize_face(emb, db, threshold)
        results.append({
            "bbox": face["bbox"],
            "name": name,
            "score": score,
        })

    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        color = COLOR_KNOWN if r["name"] != "Unknown" else COLOR_UNKNOWN
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{r['name']} ({r['score']:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return frame


def detect_from_image(image_path: str, detector: FaceDetector,
                      embedder: FaceEmbedder, db: FaceDatabase,
                      threshold: float, output_path: str | None = None):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Tidak bisa membaca gambar: {image_path}")
        return

    result = process_frame(frame, detector, embedder, db, threshold)
    cv2.imshow("Deteksi Wajah", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if output_path:
        cv2.imwrite(output_path, result)
        print(f"[INFO] Hasil disimpan ke: {output_path}")


def detect_from_webcam(detector: FaceDetector, embedder: FaceEmbedder,
                       db: FaceDatabase, threshold: float):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Tidak bisa membuka webcam.")
        return

    print("[INFO] Tekan ESC untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = process_frame(frame, detector, embedder, db, threshold)
        cv2.imshow("Deteksi Wajah - Webcam", result)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Deteksi dan kenali wajah")
    parser.add_argument("--image", "-i", help="Path gambar (kosongkan untuk webcam)")
    parser.add_argument("--model", default="yolov8n.pt", help="Path model YOLO")
    parser.add_argument("--database", default="face_database.json",
                        help="Path database wajah")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"Threshold similarity (default: {THRESHOLD})")
    parser.add_argument("--output", "-o", help="Path output gambar hasil deteksi")
    args = parser.parse_args()

    db = FaceDatabase(db_path=args.database)
    if db.count() == 0:
        print("[WARN] Database kosong. Daftarkan wajah terlebih dahulu "
              "dengan: python register.py <nama>")
        sys.exit(1)

    print(f"[INFO] Database: {db.count()} orang terdaftar.")
    print("[INFO] Loading models...")
    detector = FaceDetector(model_path=args.model)
    embedder = FaceEmbedder()

    if args.image:
        detect_from_image(args.image, detector, embedder, db,
                          args.threshold, args.output)
    else:
        detect_from_webcam(detector, embedder, db, args.threshold)


if __name__ == "__main__":
    main()
