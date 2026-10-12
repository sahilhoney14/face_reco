"""
Camera Diagnostic & Test Utility
Run: python test_camera.py
"""
import cv2
import sys

def scan_cameras():
    print("=" * 50)
    print("  SCANNING FOR CONNECTED CAMERAS")
    print("=" * 50)

    working_cameras = []

    for index in range(5):
        # Try DirectShow first on Windows, then default
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)

        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                working_cameras.append((index, w, h))
                print(f" [OK] Camera found at index {index} (Resolution: {w}x{h})")

    print("-" * 50)
    if working_cameras:
        print(f" Success! Found {len(working_cameras)} working camera(s).")
        print(f" To use this camera, run:")
        print(f"   python detect.py --camera {working_cameras[0][0]}")
        print(f"   python register.py \"Your Name\" --camera {working_cameras[0][0]}")
    else:
        print(" No working camera detected on index 0 to 4.")
        print("\nPossible solutions:")
        print(" 1. Check Windows Privacy Settings:")
        print("    Settings > Privacy & security > Camera")
        print("    -> Turn ON 'Camera access'")
        print("    -> Turn ON 'Let desktop apps access your camera'")
        print(" 2. If using a laptop, check if the physical webcam shutter is closed")
        print("    or if a function key toggles it (e.g. F8, F10, Fn+F6).")
        print(" 3. Use your Smartphone as a Webcam (DroidCam / Iriun Webcam).")
    print("=" * 50)

    return working_cameras

def preview_camera(index=0):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        print(f"Cannot open camera {index} for preview.")
        return

    print(f"\nOpening live preview for camera {index}...")
    print("Press ESC or 'q' on the preview window to close.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(f"Camera {index} Preview (Press ESC to close)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cams = scan_cameras()
    if cams:
        print(f"\nOpening preview for camera {cams[0][0]}...")
        preview_camera(cams[0][0])
