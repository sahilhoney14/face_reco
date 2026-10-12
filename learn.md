# Learn — Face Detection & Recognition

Learning documentation for the **Ai-vision** project: a face detection and recognition system built using **YOLOv8** + **InsightFace** (ArcFace).

---

## Table of Contents

1. [What is this Project?](#1-what-is-this-project)
2. [Technologies Used](#2-technologies-used)
3. [Architecture & Pipeline](#3-architecture--pipeline)
4. [Deep Dive: Component by Component](#4-deep-dive-component-by-component)
   - 4.1 FaceDetector (YOLO)
   - 4.2 FaceEmbedder (InsightFace / ArcFace)
   - 4.3 FaceDatabase (JSON Storage)
   - 4.4 Recognition Logic (Cosine Similarity)
5. [Complete Workflow](#5-complete-workflow)
6. [CLI Reference](#6-cli-reference)
7. [Compute & Performance Analysis](#7-compute--performance-analysis)
8. [Concepts & Terminology](#8-concepts--terminology)
9. [Ideas for Future Development](#9-ideas-for-future-development)

---

## 1. What is this Project?

This project builds a straightforward **face recognition** system capable of:

- **Detecting faces** from images or webcams in real time
- **Registering new faces** into a local database
- **Recognizing known faces** based on embedding vector similarity

The core pipeline:

```
Input (Image / Webcam)
  → YOLOv8 detects face bounding boxes
  → InsightFace ArcFace generates embeddings (512-dimensional vector)
  → Cosine Similarity compares against registered database
  → Output: Name + Confidence score
```

---

## 2. Technologies Used

| Technology | Role | Why Chosen |
|---|---|---|
| **YOLOv8n** | Face detection (bounding boxes) | Fast, lightweight (3.2M params), ideal for real time |
| **InsightFace buffalo_l** | Face embedding (ArcFace) | State-of-the-art accuracy, produces 512-dim feature vectors |
| **ONNX Runtime** | Inference engine | Cross-platform, supports both GPU (CUDA) and CPU |
| **OpenCV** | Image processing & display | Industry standard, robust and easy to use |
| **NumPy** | Vector & matrix math | Fast embedding and similarity calculations |
| **scikit-learn** | Utility metrics | Optional, used for evaluation |

### Why use two separate models?

YOLO and InsightFace handle distinct tasks:
- **YOLO** answers *"Where is the face?"* → produces bounding box `[x1, y1, x2, y2]`
- **InsightFace** answers *"Who is the face?"* → produces a 512-dimensional embedding vector

Separating detection from embedding keeps the pipeline flexible, modular, and easy to optimize.

---

## 3. Architecture & Pipeline

### Data Flow

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Input Frame │────▶│  YOLO Detection   │────▶│ InsightFace      │────▶│ Cosine Sim   │
│  (BGR image) │     │  (bounding box)   │     │ ArcFace Embed    │     │ vs Database  │
└─────────────┘     └──────────────────┘     └─────────────────┘     └──────────────┘
                           │                        │                        │
                     box coordinates          512-dim vector           similarity score
                           │                        │                        │
                           ▼                        ▼                        ▼
                     face_crop (ROI)          embedding vector         "Name" / "Unknown"
```

### File Structure

```
FACE_DETECTION/
├── detect.py              # Main entry point: detection + recognition
├── register.py            # Main entry point: face enrollment
├── face_database.json     # Stored face embeddings (JSON database)
├── yolov8n.pt             # YOLOv8 nano model weights
├── pyproject.toml         # Dependencies & project metadata
├── registered_faces/      # Optional folder for saved face crops
├── src/Ai-vision/
│   └── __init__.py
└── utils/
    ├── __init__.py        # Exports: FaceDetector, FaceEmbedder, FaceDatabase
    ├── detector.py        # FaceDetector — YOLOv8 face detection
    ├── embedding.py       # FaceEmbedder — InsightFace ArcFace embeddings
    └── database.py        # FaceDatabase — JSON-based face storage
```

---

## 4. Deep Dive: Component by Component

### 4.1 FaceDetector (`utils/detector.py`)

**Task:** Accept a BGR image frame and return a list of detected faces with bounding boxes.

```python
class FaceDetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_faces(self, frame: np.ndarray) -> list[dict]:
        # ...
```

**Workflow:**
1. Load the YOLOv8 nano model weights.
2. Pass the frame into YOLO → get all detection boxes.
3. Filter for class `0` (person) with confidence ≥ threshold.
4. Safely clamp coordinates and crop the face region from the frame.
5. Return a list of dictionaries: `{ bbox, confidence, face_crop }`.

**Why YOLOv8n?**
- `n` = nano, the smallest model variant → 3.2M parameters, ~8.7 GFLOPs.
- Fast enough to run on standard CPUs (~5–12 FPS) and hundreds of FPS on GPUs.

**Output format:**
```python
[
    {
        "bbox": (x1, y1, x2, y2),   # Pixel coordinates
        "confidence": 0.87,          # YOLO confidence score
        "face_crop": np.ndarray      # Cropped face image ROI
    },
    ...
]
```

---

### 4.2 FaceEmbedder (`utils/embedding.py`)

**Task:** Accept a cropped face image and output a 512-dimensional embedding vector.

```python
class FaceEmbedder:
    def __init__(self, model_name="buffalo_l", det_size=(640, 640)):
        # Dynamically selects CUDAExecutionProvider or CPUExecutionProvider
        # ...
```

**Workflow:**
1. Load InsightFace `buffalo_l` model bundle (RetinaFace/SCRFD + ArcFace).
2. Convert image crop from BGR to RGB (InsightFace expects RGB).
3. Feed crop to ArcFace to extract facial landmarks and identity features.
4. If multiple faces exist in the crop, select the one with highest detection score.
5. If crop borders are too tight for SCRFD, apply automatic border padding.
6. Return the 512-dimensional embedding vector (or `None` if unreadable).

**What is `buffalo_l`?**
- `buffalo_l` is InsightFace's comprehensive model bundle:
  - **SCRFD / RetinaFace** → Face detection & landmark localization.
  - **ArcFace** → Identity feature extraction.
- `l` = large: the highest-accuracy version.

**Why 512 dimensions?**
- ArcFace maps facial features onto a 512-dimensional hypersphere.
- Faces belonging to the same person produce vectors with small angular distance (high cosine similarity).

---

### 4.3 FaceDatabase (`utils/database.py`)

**Task:** Persist and retrieve face embedding vectors in JSON format.

```python
class FaceDatabase:
    def __init__(self, db_path="face_database.json"):
        self.db_path = db_path
        self.data: dict[str, list[list[float]]] = {}
        self._load()
```

**JSON Structure:**
```json
{
  "John": [
    [-0.827, -0.632, ...],   // Embedding vector 1 (512 floats)
    [-0.986, -0.339, ...],   // Embedding vector 2
    [-0.743, 1.169, ...]     // Embedding vector 3
  ],
  "Sarah": [
    [0.123, -0.456, ...]
  ]
}
```

**Methods:**
- `register(name, embedding)` — Append a new embedding vector for a person.
- `get_all_embeddings()` — Retrieve all registered persons as `dict[str, list[np.ndarray]]`.
- `remove(name)` — Delete a person's entries.
- `count()` — Count registered persons.

**Why store multiple embeddings per person?**
- Enrolling multiple photos (different angles, lighting, facial expressions) substantially improves recognition accuracy under real-world conditions.

---

### 4.4 Recognition Logic (`detect.py`)

**Task:** Compare extracted face embeddings against registered embeddings using Cosine Similarity.

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def recognize_face(embedding, db, threshold=0.6):
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
```

**Workflow:**
1. Extract embedding vector from the detected face.
2. Compare against every stored embedding vector in the database.
3. Track the highest similarity score and corresponding name.
4. If `score >= threshold` (default `0.6`), label as identified name; otherwise mark as **"Unknown"**.

**What is Cosine Similarity?**
- Measures the cosine of the angle between two vectors in 512-dimensional space.
- Ranges from -1 to 1:
  - **1** = Identical direction (0° angle)
  - **0** = Orthogonal / completely unrelated (90° angle)
  - **-1** = Opposite direction (180° angle)
- In face recognition, a score ≥ `0.6` typically indicates a reliable identity match.

---

## 5. Complete Workflow

### Registering a Face (`register.py`)

```powershell
python register.py "John Doe" --image photo.jpg
```
1. Load YOLOv8 and InsightFace models.
2. Load/create `face_database.json`.
3. Read image → detect face with YOLO.
4. Crop face ROI → generate 512-dim ArcFace embedding.
5. Append embedding vector to `data["John Doe"]` and write to JSON.

### Recognizing Faces in an Image (`detect.py`)

```powershell
python detect.py --image photo.jpg --output result.png
```
1. Load models and face database.
2. Read image frame.
3. Detect faces using YOLO → `[ {bbox, face_crop}, ... ]`.
4. Generate embeddings with ArcFace.
5. Compare against database using cosine similarity.
6. Draw bounding boxes (green for recognized, red for unknown) and identity labels.
7. Save output and display result window.

### Real-Time Webcam Stream (`detect.py`)

```powershell
python detect.py
```
1. Open webcam stream (`cv2.VideoCapture`).
2. Continuous loop: read frame → run detection & recognition → render overlay.
3. Press `ESC` to exit.

---

## 6. CLI Reference

### `register.py`

```powershell
# Register from an image file
python register.py "Name" --image path/to/photo.jpg

# Register interactively via webcam (press SPACE to capture)
python register.py "Name"

# Specify custom webcam index or model path
python register.py "Name" --camera 1 --database custom_db.json
```

| Argument | Default | Description |
|---|---|---|
| `name` | *(required)* | Name of the person to enroll |
| `--image`, `-i` | `None` (webcam) | Path to image file |
| `--camera`, `-c` | `0` | Webcam index (0, 1) or stream URL |
| `--model` | `yolov8n.pt` | Path to YOLO model weights |
| `--database` | `face_database.json` | Path to face database file |

### `detect.py`

```powershell
# Detect from image
python detect.py --image photo.jpg

# Save output to image file
python detect.py --image photo.jpg --output result.png

# Headless / batch mode (no GUI window popup)
python detect.py --image photo.jpg --output result.png --no-show

# Live webcam recognition
python detect.py

# Custom webcam index or similarity threshold
python detect.py --camera 1 --threshold 0.65
```

| Argument | Default | Description |
|---|---|---|
| `--image`, `-i` | `None` (webcam) | Path to image file |
| `--camera`, `-c` | `0` | Webcam index (0, 1) or stream URL |
| `--model` | `yolov8n.pt` | Path to YOLO model weights |
| `--database` | `face_database.json` | Path to face database file |
| `--threshold` | `0.6` | Cosine similarity match threshold |
| `--output`, `-o` | `None` | Path to save annotated output image |
| `--no-show` | `False` | Do not display GUI pop-up window |

---

## 7. Compute & Performance Analysis

### Pipeline Compute Cost

| Stage | Model | Compute | Notes |
|---|---|---|---|
| YOLO Detection | YOLOv8n (3.2M params) | **~8.7 GFLOPs** | Evaluated 1x per frame |
| InsightFace Embedding | buffalo_l | **~0.5–1.0 GFLOPs** per face | Evaluated on small face crops |
| Cosine Similarity | Pure NumPy | **~1K FLOPs** per comparison | Negligible vector dot product |

### Database Size Scaling Impact

| Component | 1 face, DB=1 | 1 face, DB=1000 | Delta |
|---|---|---|---|
| YOLO Detection | 8.7 GFLOPs | 8.7 GFLOPs | 0% |
| InsightFace Embedding | ~0.75 GFLOPs | ~0.75 GFLOPs | 0% |
| Cosine Similarity | ~1 KFLOPs | ~1 MFLOPs | Negligible |
| **Total** | **~9.45 GFLOPs** | **~9.45 GFLOPs** | **< 0.01%** |

> **Takeaway:** Database size has virtually no compute impact on modern hardware. The primary bottlenecks are YOLO detection and ArcFace feature extraction.

### Memory Footprint

| Component | Approximate Memory |
|---|---|
| YOLOv8n model | ~6.4 MB (FP16) |
| buffalo_l model bundle | ~300 MB |
| Database (1 embedding) | ~4 KB |
| Database (1000 embeddings) | ~4 MB |

---

## 8. Concepts & Terminology

### YOLO (You Only Look Once)
- Single-shot object detection neural network.
- YOLOv8n is the nano variant, optimized for edge devices and high frame rates.

### InsightFace
- Leading open-source framework for 2D/3D deep face analysis.
- Combines RetinaFace / SCRFD detectors and ArcFace feature extractors.

### ArcFace
- Additive Angular Margin Loss for deep face recognition.
- Maximizes face feature margin on the angular hypersphere to ensure high inter-class variance and low intra-class variance.

### Cosine Similarity
- Compares directional alignment of two unit vectors: `dot(A, B) / (norm(A) * norm(B))`.
- Values close to `1.0` indicate a match; values near `0.0` indicate different identities.

---

## 9. Ideas for Future Extensions

- **[ ] Face verification**: 1-to-1 face comparison.
- **[ ] Liveness / Anti-spoofing**: Distinguish live faces from printed photos or screens.
- **[ ] Web Interface**: FastAPI / Flask dashboard for browser-based monitoring.
- **[ ] Video file processing**: Process full `.mp4` video recordings.
- **[ ] Vector Database**: Integrate FAISS or Chroma for sub-millisecond search over 100,000+ faces.
