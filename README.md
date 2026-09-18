# Face-Detection

Face detection and recognition system using **YOLOv8** + **InsightFace** (RetinaFace + ArcFace). Detects faces from images or webcam in real-time, then identifies them against a registered face database using cosine similarity on 512-dimensional embeddings.

## Features

- **Real-time face detection** — YOLOv8n for fast, lightweight face bounding box detection
- **Face recognition** — InsightFace ArcFace embeddings for identity matching
- **Register from image or webcam** — easily enroll new faces with a single command
- **Live webcam recognition** — real-time identification on video stream
- **JSON-based face database** — simple, portable, and human-readable storage
- **Configurable similarity threshold** — tune precision vs recall tradeoff

## Project Structure

```
Ai-vision/
├── detect.py              # Main detection/recognition script
├── register.py            # Face registration script
├── face_database.json     # Face embedding database (auto-created)
├── yolov8n.pt             # YOLOv8 nano model (auto-downloaded)
├── pyproject.toml         # Project metadata & dependencies
├── registered_faces/      # Directory for registered face images
├── src/Ai-vision/
│   └── __init__.py
└── utils/
    ├── __init__.py
    ├── detector.py        # FaceDetector — YOLOv8 face detection
    ├── embedding.py       # FaceEmbedder — InsightFace ArcFace embeddings
    └── database.py        # FaceDatabase — JSON-based face storage
```

## Requirements

- Python 3.13
- GPU recommended (CUDA) for real-time performance
- Works on CPU with reduced FPS

## Installation

```bash
# Clone the repository
git clone https://github.com/frdiskandar/Ai-vision.git
cd Ai-vision

# Create virtual environment and install dependencies
uv sync
```

Or with pip:

```bash
pip install ultralytics insightface onnxruntime opencv-python numpy scikit-learn
```

## Usage

### Register a New Face

From an image:

```bash
python register.py "Farid" --image photos/farid.jpg
```

From webcam (press SPACE to capture):

```bash
python register.py "Farid"
```

### Detect and Recognize

From an image:

```bash
python detect.py --image photo.jpg
```

Save output to file:

```bash
python detect.py --image photo.jpg --output result.jpg
```

From webcam (press ESC to exit):

```bash
python detect.py
```

With custom settings:

```bash
python detect.py --image photo.jpg --threshold 0.5 --database custom_db.json
```

### CLI Options

**register.py**

| Argument | Description |
|----------|-------------|
| `name` | Name of the person to register |
| `--image`, `-i` | Path to image file (omit for webcam) |
| `--model` | Path to YOLO model (default: `yolov8n.pt`) |
| `--database` | Path to face database (default: `face_database.json`) |

**detect.py**

| Argument | Description |
|----------|-------------|
| `--image`, `-i` | Path to image file (omit for webcam) |
| `--model` | Path to YOLO model (default: `yolov8n.pt`) |
| `--database` | Path to face database (default: `face_database.json`) |
| `--threshold` | Similarity threshold (default: `0.6`) |
| `--output`, `-o` | Save annotated output image |

## How It Works

### Pipeline

```
Frame → [1] YOLO Detection → [2] InsightFace Embedding (per face) → [3] Cosine Similarity (per face)
```

1. **YOLO Detection** — YOLOv8n detects all faces and returns bounding boxes
2. **InsightFace Embedding** — Each face crop is passed through ArcFace to generate a 512-dim vector
3. **Cosine Similarity** — Each embedding is compared against all registered embeddings in the database

### Matching Logic

- A face is recognized as `Known` if the highest cosine similarity score ≥ threshold (default 0.6)
- Otherwise it is labeled `Unknown`
- Multiple embeddings per person are supported (improves accuracy with varied angles/lighting)

---

## Compute Analysis

### Pipeline Inference

```
Frame → [1] YOLO Detection → [2] InsightFace Embedding (per wajah) → [3] Cosine Similarity (per wajah)
```

| Tahap | Model | Compute | Catatan |
|-------|-------|---------|---------|
| YOLO Detection | YOLOv8n (3.2M params) | **~8.7 GFLOPs** | 1x per frame, deteksi semua wajah |
| InsightFace Embedding | buffalo_l (RetinaFace + ArcFace) | **~0.5–1.0 GFLOPs per wajah** | Input crop kecil ~100-300px |
| Cosine Similarity | — | **~1K FLOPs per perbandingan** | Vektor 512-dim, hampir nihil |

### Perbandingan: DB 1 vs DB 1000

| Komponen | 1 face, DB=1 | 1 face, DB=1000 | Delta |
|----------|-------------|-----------------|-------|
| YOLO Detection | 8.7 GFLOPs | 8.7 GFLOPs | 0% |
| InsightFace Embedding | ~0.75 GFLOPs | ~0.75 GFLOPs | 0% |
| Cosine Similarity | ~1K FLOPs | ~1 MFLOPs | Naik tapi negligible |
| **Total** | **~9.45 GFLOPs** | **~9.45 GFLOPs** | **~0.001%** |

**Database size hampir tidak mempengaruhi compute.** Bottleneck selalu YOLO + InsightFace embedding.

### Memory

| Komponen | Ukuran |
|----------|--------|
| YOLOv8n model | ~6.4 MB (FP16) |
| buffalo_l (full) | ~300 MB |
| DB 1 embedding | 4 KB |
| DB 1000 embeddings | 4 MB |

### Estimated Inference Time (per frame, 1 wajah terdeteksi)

| Hardware | Waktu/frame | FPS | Catatan |
|----------|-------------|-----|---------|
| GPU (RTX 4060) | ~3–7 ms | ~140–300 | CUDA, optimal |
| GPU (RTX 3060) | ~5–10 ms | ~100–200 | CUDA |
| CPU (Ryzen 5 6600H) | ~110–180 ms | ~5–9 | ONNX Runtime CPU |
| CPU (i7 12th gen) | ~80–120 ms | ~8–12 | ONNX Runtime CPU |
| CPU (i5 10th gen) | ~150–250 ms | ~4–7 | ONNX Runtime CPU |

---

## License

MIT
# Face-detection
