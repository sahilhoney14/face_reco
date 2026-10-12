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
                      threshold: float, output_path: str | None = None,
                      show: bool = True):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Unable to read image: {image_path}")
        return

    result = process_frame(frame, detector, embedder, db, threshold)

    if output_path:
        cv2.imwrite(output_path, result)
        print(f"[INFO] Result saved to: {output_path}")

    if show:
        try:
            print("[INFO] Displaying result. Press any key on the image window to close...")
            cv2.imshow("Face Detection", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as e:
            print(f"[WARN] Unable to open GUI window: {e}")


def detect_from_webcam(detector: FaceDetector, embedder: FaceEmbedder,
                       db: FaceDatabase, threshold: float, camera_source: str = "0"):
    # Try parsing camera_source as an integer device index; otherwise treat as stream URL / file
    try:
        cam_id = int(camera_source)
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_id)
    except ValueError:
        cam_id = camera_source
        cap = cv2.VideoCapture(cam_id)

    if not cap.isOpened():
        print(f"[ERROR] Unable to open webcam (source: {camera_source}).")
        print("\n[WEBCAM CONFIGURATION GUIDE]:")
        print("1. Ensure webcam is plugged in and not in use by other apps (Zoom, Teams, etc.).")
        print("2. Windows Camera Permission: Settings > Privacy & security > Camera, turn ON 'Let desktop apps access your camera'.")
        print("3. If you have an external or second camera, try a different index:")
        print("   python detect.py --camera 1")
        print("4. If using your phone as a camera via Wi-Fi (DroidCam / Iriun / IP Webcam):")
        print("   python detect.py --camera http://192.168.1.X:8080/video")
        return

    print("[INFO] Webcam started successfully. Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = process_frame(frame, detector, embedder, db, threshold)
        cv2.imshow("Face Detection - Webcam", result)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Detect and recognize faces")
    parser.add_argument("--image", "-i", help="Path to image file (leave empty for webcam)")
    parser.add_argument("--camera", "-c", default="0",
                        help="Webcam index (0, 1, 2) or stream URL (default: 0)")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--database", default="face_database.json",
                        help="Path to face database JSON file")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"Similarity threshold (default: {THRESHOLD})")
    parser.add_argument("--output", "-o", help="Output path for annotated image")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not display GUI window (useful for CLI/headless runs)")
    args = parser.parse_args()

    db = FaceDatabase(db_path=args.database)
    if db.count() == 0:
        print("[WARN] Database is empty. Please register a face first "
              "with: python register.py <name>")
        sys.exit(1)

    print(f"[INFO] Database: {db.count()} person(s) registered.")
    print("[INFO] Loading models...")
    detector = FaceDetector(model_path=args.model)
    embedder = FaceEmbedder()

    if args.image:
        detect_from_image(args.image, detector, embedder, db,
                          args.threshold, args.output, show=not args.no_show)
    else:
        detect_from_webcam(detector, embedder, db, args.threshold, camera_source=args.camera)


if __name__ == "__main__":
    main()
