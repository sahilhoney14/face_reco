# Learn — Face Detection & Recognition

Dokumentasi pembelajaran untuk project **Ai-vision**, sistem deteksi dan pengenalan wajah menggunakan **YOLOv8** + **InsightFace** (ArcFace).

---

## Table of Contents

1. [Apa itu Project ini?](#1-apa-itu-project-ini)
2. [Teknologi yang Digunakan](#2-teknologi-yang-digunakan)
3. [Arsitektur & Pipeline](#3-arsitektur--pipeline)
4. [Deep Dive: Component per Component](#4-deep-dive-component-per-component)
   - 4.1 FaceDetector (YOLO)
   - 4.2 FaceEmbedder (InsightFace / ArcFace)
   - 4.3 FaceDatabase (JSON Storage)
   - 4.4 Recognition Logic (Cosine Similarity)
5. [Alur Kerja Lengkap](#5-alur-kerja-lengkap)
6. [CLI Reference](#6-cli-reference)
7. [Compute & Performance Analysis](#7-compute--performance-analysis)
8. [Concepts & Terminology](#8-concepts--terminology)
9. [Ide untuk Pengembangan Selanjutnya](#9-ide-untuk-pengembangan-selanjutnya)

---

## 1. Apa itu Project ini?

Project ini membangun sistem **face recognition** sederhana yang bisa:

- **Mendeteksi wajah** dari gambar atau webcam secara real-time
- **Mendaftarkan wajah** baru ke dalam database
- **Mengenali wajah** yang sudah terdaftar berdasarkan kemiripan vektor embedding

Pipeline utama:

```
Input (gambar/webcam)
  → YOLOv8 deteksi bounding box wajah
  → InsightFace ArcFace generate embedding (vektor 512-dim)
  → Cosine Similarity dibandingkan ke database
  → Output: nama wajah + confidence score
```

---

## 2. Teknologi yang Digunakan

| Teknologi | Fungsi | Kenapa Dipilih |
|-----------|--------|----------------|
| **YOLOv8n** | Deteksi wajah (bounding box) | Cepat, ringan (3.2M params), cocok untuk real-time |
| **InsightFace buffalo_l** | Face embedding (ArcFace) | State-of-the-art akurasi, menghasilkan vektor 512-dim |
| **ONNX Runtime** | Inference engine | Cross-platform, mendukung GPU (CUDA) dan CPU |
| **OpenCV** | Image processing & display | Standar industry, mudah digunakan |
| **NumPy** | Operasi vektor & matriks | Komputasi embedding & similarity |
| **scikit-learn** | Utility metrics | Opsional, untuk evaluasi model |

### Kenapa dua model terpisah?

YOLO dan InsightFace punya tugas berbeda:
- **YOLO** = "Di mana wajahnya?" → menghasilkan bounding box `[x1, y1, x2, y2]`
- **InsightFace** = "Siapa wajahnya?" → menghasilkan vektor embedding 512-dim

Pisahkan detection dan embedding supaya lebih fleksibel dan modular.

---

## 3. Arsitektur & Pipeline

### Alur Data

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Input Frame │────▶│  YOLO Detection   │────▶│ InsightFace      │────▶│ Cosine Sim   │
│  (BGR image) │     │  (bounding box)   │     │ ArcFace Embed    │     │ vs Database  │
└─────────────┘     └──────────────────┘     └─────────────────┘     └──────────────┘
                           │                        │                        │
                     box coordinates          512-dim vector           similarity score
                           │                        │                        │
                           ▼                        ▼                        ▼
                     face_crop (ROI)          embedding vector         "Farid" / "Unknown"
```

### Struktur File

```
Ai-vision/
├── detect.py              # Entry point: deteksi + recognition
├── register.py            # Entry point: registrasi wajah baru
├── face_database.json     # Database embedding wajah (auto-created)
├── yolov8n.pt             # Model YOLOv8 nano (auto-downloaded)
├── pyproject.toml         # Dependencies
├── registered_faces/      # Folder untuk gambar wajah terdaftar
├── src/Ai-vision/
│   └── __init__.py
└── utils/
    ├── __init__.py         # Export: FaceDetector, FaceEmbedder, FaceDatabase
    ├── detector.py         # FaceDetector — YOLOv8 face detection
    ├── embedding.py        # FaceEmbedder — InsightFace ArcFace embeddings
    └── database.py         # FaceDatabase — JSON-based face storage
```

---

## 4. Deep Dive: Component per Component

### 4.1 FaceDetector (`utils/detector.py`)

**Tugas:** Menerima gambar (BGR), menghasilkan list wajah dengan bounding box.

```python
class FaceDetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_faces(self, frame: np.ndarray) -> list[dict]:
        # ...
```

**Alur kerja:**
1. Load model YOLOv8 nano
2. Feed frame ke model → dapatkan semua bounding box
3. Filter hanya class `0` (person/wajah) dengan confidence ≥ threshold
4. Crop wajah dari frame berdasarkan bounding box
5. Return list of dict: `{ bbox, confidence, face_crop }`

**Kenapa pakai YOLOv8n?**
- `n` = nano, ukuran paling kecil → 3.2M params, ~8.7 GFLOPs
- Cukup untuk deteksi wajah, tidak perlu model besar
- Bisa jalan di CPU dengan FPS yang bisa diterima (~5-12 FPS)

**Output format:**
```python
[
    {
        "bbox": (x1, y1, x2, y2),   # koordinat pixel
        "confidence": 0.87,          # confidence score YOLO
        "face_crop": np.ndarray      # ROI wajah yang sudah di-crop
    },
    ...
]
```

---

### 4.2 FaceEmbedder (`utils/embedding.py`)

**Tugas:** Menerima crop wajah, menghasilkan vektor embedding 512-dim.

```python
class FaceEmbedder:
    def __init__(self, model_name="buffalo_l", det_size=(640, 640)):
        self.app = FaceAnalysis(
            name=model_name,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=det_size)

    def get_embedding(self, face_crop: np.ndarray) -> np.ndarray | None:
        # ...
```

**Alur kerja:**
1. Load InsightFace `buffalo_l` model (RetinaFace untuk deteksi + ArcFace untuk embedding)
2. Konversi BGR → RGB (InsightFace butuh RGB)
3. Feed crop wajah ke ArcFace
4. Jika ada beberapa wajah terdeteksi dalam crop → ambil yang confidence tertinggi
5. Return vektor embedding 512-dim (atau None jika gagal)

**Apa itu buffalo_l?**
- `buffalo_l` adalah model pack InsightFace yang berisi:
  - **RetinaFace** → face detection (dalam crop, untuk menemukan wajah yang presisi)
  - **ArcFace** → face recognition (menghasilkan embedding)
- `l` = large, versi terbesar → akurasi terbaik

**Kenapa 512-dim?**
- ArcFace dirancang untuk menghasilkan vektor 512 dimensi
- Vektor ini merepresentasikan "identitas" wajah dalam bentuk numerik
- Vektor yang mirip (angle kecil) = wajah yang mirip

**CUDA vs CPU:**
- `providers=["CUDAExecutionProvider", "CPUExecutionProvider"]`
- Akan otomatis pakai CUDA jika tersedia, fallback ke CPU

---

### 4.3 FaceDatabase (`utils/database.py`)

**Tugas:** Menyimpan dan mengambil embedding wajah dalam format JSON.

```python
class FaceDatabase:
    def __init__(self, db_path="face_database.json"):
        self.db_path = db_path
        self.data: dict[str, list[list[float]]] = {}
        self._load()
```

**Format JSON:**
```json
{
  "Farid": [
    [-0.827, -0.632, ...],   // embedding 1 (512 float)
    [-0.986, -0.339, ...],   // embedding 2
    [-0.743, 1.169, ...]     // embedding 3
  ],
  "Budi": [
    [0.123, -0.456, ...]
  ]
}
```

**Fitur:**
- `register(name, embedding)` — tambah embedding baru (bisa multiple per orang)
- `get_all_embeddings()` — ambil semua data sebagai `dict[str, list[np.ndarray]]`
- `remove(name)` — hapus semua data untuk satu orang
- `count()` — jumlah orang terdaftar

**Kenapa multiple embedding per orang?**
- Satu orang bisa punya banyak embedding dari angle, pencahayaan, ekspresi berbeda
- Saat recognition, semua embedding dibandingkan → diambil yang paling mirip
- Ini meningkatkan akurasi secara signifikan

---

### 4.4 Recognition Logic (`detect.py`)

**Tugas:** Membandingkan embedding wajah dengan database menggunakan Cosine Similarity.

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

**Alur:**
1. Ambil embedding dari wajah yang terdeteksi
2. Bandingkan dengan **semua** embedding di database (brute-force)
3. Catat similarity tertinggi dan nama yang cocok
4. Jika score ≥ threshold (default 0.6) → **Known**, else → **Unknown**

**Apa itu Cosine Similarity?**
- Mengukur sudut antara dua vektor dalam ruang 512-dim
- Nilai antara -1 sampai 1:
  - **1** = identik (sudut 0°)
  - **0** = tidak ada kemiripan (sudut 90°)
  - **-1** = kebalikan (sudut 180°)
- Untuk face recognition, score ≥ 0.6 dianggap match

---

## 5. Alur Kerja Lengkap

### Registrasi Wajah (`register.py`)

```
python register.py "Farid" --image photo.jpg

1. Load YOLOv8 model
2. Load InsightFace buffalo_l model
3. Load/create face_database.json
4. Baca gambar → detect wajah dengan YOLO
5. Untuk setiap wajah:
   a. Crop wajah
   b. Generate embedding 512-dim dengan ArcFace
   c. Simpan ke database: data["Farid"].append(embedding)
6. Simpan database ke JSON
```

### Deteksi & Recognition (`detect.py`)

```
python detect.py --image photo.jpg

1. Load semua models
2. Load database wajah
3. Baca gambar
4. Detect semua wajah dengan YOLO → list of {bbox, face_crop}
5. Untuk setiap wajah:
   a. Generate embedding dengan ArcFace
   b. Bandingkan dengan semua embedding di database (cosine similarity)
   c. Jika score ≥ 0.6 → tampilkan nama, else → "Unknown"
6. Gambar bounding box + label di atas frame
7. Tampilkan hasil / simpan ke file
```

### Webcam Real-time

```
python detect.py

1. Buka webcam (VideoCapture(0))
2. Loop:
   a. Baca frame
   b. Jalankan pipeline deteksi + recognition
   c. Tampilkan frame dengan overlay
   d. Tekan ESC untuk keluar
```

---

## 6. CLI Reference

### register.py

```bash
# Registrasi dari gambar
python register.py "Nama" --image path/to/foto.jpg

# Registrasi dari webcam (tekan SPACE untuk capture)
python register.py "Nama"

# Dengan opsi lain
python register.py "Nama" --image foto.jpg --model yolov8n.pt --database db.json
```

| Argument | Default | Deskripsi |
|----------|---------|-----------|
| `name` | (required) | Nama orang yang akan didaftarkan |
| `--image`, `-i` | None (webcam) | Path ke file gambar |
| `--model` | `yolov8n.pt` | Path model YOLO |
| `--database` | `face_database.json` | Path database wajah |

### detect.py

```bash
# Deteksi dari gambar
python detect.py --image photo.jpg

# Simpan hasil ke file
python detect.py --image photo.jpg --output result.jpg

# Webcam real-time
python detect.py

# Threshold custom
python detect.py --image photo.jpg --threshold 0.7
```

| Argument | Default | Deskripsi |
|----------|---------|-----------|
| `--image`, `-i` | None (webcam) | Path ke file gambar |
| `--model` | `yolov8n.pt` | Path model YOLO |
| `--database` | `face_database.json` | Path database wajah |
| `--threshold` | `0.6` | Threshold cosine similarity |
| `--output`, `-o` | None | Path output gambar hasil deteksi |

---

## 7. Compute & Performance Analysis

### Pipeline Compute

| Tahap | Model | Compute | Catatan |
|-------|-------|---------|---------|
| YOLO Detection | YOLOv8n (3.2M params) | **~8.7 GFLOPs** | 1x per frame |
| InsightFace Embedding | buffalo_l | **~0.5–1.0 GFLOPs** per wajah | Input crop kecil ~100-300px |
| Cosine Similarity | — | **~1K FLOPs** per perbandingan | Vektor 512-dim, hampir nihil |

### Database Size Impact

| Komponen | 1 face, DB=1 | 1 face, DB=1000 | Delta |
|----------|-------------|-----------------|-------|
| YOLO Detection | 8.7 GFLOPs | 8.7 GFLOPs | 0% |
| InsightFace Embedding | ~0.75 GFLOPs | ~0.75 GFLOPs | 0% |
| Cosine Similarity | ~1K FLOPs | ~1 MFLOPs | Negligible |
| **Total** | **~9.45 GFLOPs** | **~9.45 GFLOPs** | **~0.001%** |

> **Kesimpulan: Database size hampir tidak mempengaruhi compute.** Bottleneck selalu YOLO + InsightFace embedding.

### Memory

| Komponen | Ukuran |
|----------|--------|
| YOLOv8n model | ~6.4 MB (FP16) |
| buffalo_l (full) | ~300 MB |
| DB 1 embedding | 4 KB |
| DB 1000 embeddings | 4 MB |

### Estimated Inference Time (per frame, 1 wajah)

| Hardware | Waktu/frame | FPS | Catatan |
|----------|-------------|-----|---------|
| GPU (RTX 4060) | ~3–7 ms | ~140–300 | CUDA, optimal |
| GPU (RTX 3060) | ~5–10 ms | ~100–200 | CUDA |
| CPU (Ryzen 5 6600H) | ~110–180 ms | ~5–9 | ONNX Runtime CPU |
| CPU (i7 12th gen) | ~80–120 ms | ~8–12 | ONNX Runtime CPU |
| CPU (i5 10th gen) | ~150–250 ms | ~4–7 | ONNX Runtime CPU |

---

## 8. Concepts & Terminology

### YOLO (You Only Look Once)
- Model deteksi objek single-shot
- YOLOv8n = versi nano, paling ringan
- Output: bounding box + class label + confidence score
- Class `0` pada COCO dataset = `person`

### InsightFace
- Framework open-source untuk face analysis
- Menggabungkan deteksi (RetinaFace) dan recognition (ArcFace)
- Model `buffalo_l` = paket lengkap (deteksi + embedding)

### ArcFace
- Algoritma face recognition berbasis deep learning
- Training menggunakan Additive Angular Margin Loss
- Menghasilkan embedding yang diskriminatif dalam ruang hypersphere

### Embedding
- Vektor numerik berdimensi tinggi (512-dim dalam kasus ini)
- Merepresentasikan karakteristik wajah dalam bentuk angka
- Wajah yang mirip → embedding yang mirip (cosine similarity tinggi)

### Cosine Similarity
- Mengukur kemiripan antara dua vektor berdasarkan sudutnya
- Formula: `similarity = (A · B) / (||A|| × ||B||)`
- Range: -1 (berlawanan) sampai 1 (identik)
- Threshold 0.6 = batas "match" dalam project ini

### ONNX Runtime
- Framework inference untuk model ONNX
- Mendukung berbagai backend: CUDA, TensorRT, CPU
- Digunakan oleh both YOLO dan InsightFace

---

## 9. Ide untuk Pengembangan Selanjutnya

### Fitur yang bisa ditambahkan:

- **[ ] Face verification** — bandingkan 2 wajah (1:1 matching)
- **[ ] Anti-spoofing** — deteksi foto/ganti wajah (liveness detection)
- **[ ] Web interface** — tambahkan Flask/FastAPI untuk akses via browser
- **[ ] Video file processing** — deteksi dari file video (.mp4)
- **[ ] Export database** — backup/restore database wajah
- **[ ] Multiple face management** — hapus/update individual embedding
- **[ ] Attendance system** — integrasi dengan sistem absensi
- **[ ] Logging & analytics** — catatan kapan wajah terdeteksi

### Optimasi yang bisa dilakukan:

- **[ ] Face alignment** — align wajah sebelum generate embedding (meningkatkan akurasi)
- **[ ] Batch processing** — proses multiple gambar sekaligus
- **[ ] Model quantization** — kompres model untuk inference lebih cepat
- **[ ] FAISS database** — gunakan vector database untuk pencarian lebih cepat
- **[ ] GPU memory management** — unload model saat tidak digunakan

---

*Last updated: September 2026*
