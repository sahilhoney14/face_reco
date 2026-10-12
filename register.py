import argparse
import os
import sys

import cv2

from utils import FaceDetector, FaceEmbedder, FaceDatabase


def register_from_image(name: str, image_path: str, db: FaceDatabase,
                        detector: FaceDetector, embedder: FaceEmbedder):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Unable to read image: {image_path}")
        return False

    faces = detector.detect_faces(frame)
    if not faces:
        print("[INFO] No face detected in the image.")
        return False

    print(f"[INFO] {len(faces)} face(s) detected.")

    for i, face in enumerate(faces):
        emb = embedder.get_embedding(face["face_crop"])
        if emb is None:
            print(f"[WARN] Face {i+1} failed to generate embedding, skipping.")
            continue

        db.register(name, emb)
        print(f"[OK] Face {i+1} successfully registered as '{name}' "
              f"(confidence: {face['confidence']:.2f})")

    return True


def register_from_webcam(name: str, db: FaceDatabase,
                         detector: FaceDetector, embedder: FaceEmbedder,
                         camera_source: str = "0"):
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
        print(f"   python register.py \"{name}\" --camera 1")
        print("4. If using your phone as a camera via Wi-Fi:")
        print(f"   python register.py \"{name}\" --camera http://192.168.1.X:8080/video")
        return False

    print("[INFO] Press SPACE to capture face, ESC to exit.")
    print("[INFO] Make sure your face is clearly visible to the camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect_faces(frame)
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, f"Register: {name} | SPACE=Capture | ESC=Exit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Register Face", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 32:
            if not faces:
                print("[WARN] No face detected, please try again.")
                continue

            face = faces[0]
            emb = embedder.get_embedding(face["face_crop"])
            if emb is None:
                print("[WARN] Failed to generate embedding, please try again.")
                continue

            db.register(name, emb)
            print(f"[OK] Face successfully registered as '{name}' "
                  f"(confidence: {face['confidence']:.2f})")
            print(f"[INFO] Total registrations for '{name}': "
                  f"{len(db.get_all_embeddings()[name])} face(s).")
            break

    cap.release()
    cv2.destroyAllWindows()
    return True


def main():
    parser = argparse.ArgumentParser(description="Register a new face into the database")
    parser.add_argument("name", help="Name of the person")
    parser.add_argument("--image", "-i", help="Path to image file (leave empty for webcam)")
    parser.add_argument("--camera", "-c", default="0",
                        help="Webcam index (0, 1, 2) or stream URL (default: 0)")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--database", default="face_database.json",
                        help="Path to face database JSON file")
    args = parser.parse_args()

    if not args.name.strip():
        print("[ERROR] Name cannot be empty.")
        sys.exit(1)

    name = args.name.strip()

    print("[INFO] Loading models...")
    detector = FaceDetector(model_path=args.model)
    embedder = FaceEmbedder()
    db = FaceDatabase(db_path=args.database)

    print(f"[INFO] Current database: {db.count()} person(s) registered.")

    if args.image:
        register_from_image(name, args.image, db, detector, embedder)
    else:
        register_from_webcam(name, db, detector, embedder, camera_source=args.camera)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
