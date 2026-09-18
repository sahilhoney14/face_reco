import argparse
import os
import sys

import cv2

from utils import FaceDetector, FaceEmbedder, FaceDatabase


def register_from_image(name: str, image_path: str, db: FaceDatabase,
                        detector: FaceDetector, embedder: FaceEmbedder):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Tidak bisa membaca gambar: {image_path}")
        return False

    faces = detector.detect_faces(frame)
    if not faces:
        print("[INFO] Tidak ada wajah terdeteksi di gambar.")
        return False

    print(f"[INFO] {len(faces)} wajah terdeteksi.")

    for i, face in enumerate(faces):
        emb = embedder.get_embedding(face["face_crop"])
        if emb is None:
            print(f"[WARN] Wajah {i+1} gagal generate embedding, skip.")
            continue

        db.register(name, emb)
        print(f"[OK] Wajah {i+1} berhasil didaftarkan sebagai '{name}' "
              f"(confidence: {face['confidence']:.2f})")

    return True


def register_from_webcam(name: str, db: FaceDatabase,
                         detector: FaceDetector, embedder: FaceEmbedder):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Tidak bisa membuka webcam.")
        return False

    print("[INFO] Tekan SPACE untuk ambil gambar, ESC untuk keluar.")
    print("[INFO] Pastikan wajah terlihat jelas di kamera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect_faces(frame)
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, f"Register: {name} | SPACE=ambil | ESC=keluar",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Register Face", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 32:
            if not faces:
                print("[WARN] Tidak ada wajah terdeteksi, coba lagi.")
                continue

            face = faces[0]
            emb = embedder.get_embedding(face["face_crop"])
            if emb is None:
                print("[WARN] Gagal generate embedding, coba lagi.")
                continue

            db.register(name, emb)
            print(f"[OK] Wajah berhasil didaftarkan sebagai '{name}' "
                  f"(confidence: {face['confidence']:.2f})")
            print(f"[INFO] Total registrasi '{name}': "
                  f"{len(db.get_all_embeddings()[name])} wajah.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return True


def main():
    parser = argparse.ArgumentParser(description="Daftarkan wajah baru ke database")
    parser.add_argument("name", help="Nama pemilik wajah")
    parser.add_argument("--image", "-i", help="Path gambar (kosongkan untuk webcam)")
    parser.add_argument("--model", default="yolov8n.pt", help="Path model YOLO")
    parser.add_argument("--database", default="face_database.json",
                        help="Path database wajah")
    args = parser.parse_args()

    if not args.name.strip():
        print("[ERROR] Nama tidak boleh kosong.")
        sys.exit(1)

    name = args.name.strip()

    print("[INFO] Loading models...")
    detector = FaceDetector(model_path=args.model)
    embedder = FaceEmbedder()
    db = FaceDatabase(db_path=args.database)

    print(f"[INFO] Database saat ini: {db.count()} orang terdaftar.")

    if args.image:
        register_from_image(name, args.image, db, detector, embedder)
    else:
        register_from_webcam(name, db, detector, embedder)

    print("[INFO] Selesai.")


if __name__ == "__main__":
    main()
