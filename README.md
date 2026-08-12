# Blood Cell Classification API

A production-grade **FastAPI** backend that serves a trained Keras CNN model for white blood cell type classification.

This project was built as part of a college ML Wing selection process. Rather than building a standard tutorial project, it was designed to demonstrate end-to-end technical rigor — from dataset handling and architecture choices to model debugging, backend service design, and deployment.

---

## Project Overview

- **Task**: 4-class white blood cell type classification from microscopy images.
- **Classes**: `EOSINOPHIL`, `LYMPHOCYTE`, `MONOCYTE`, `NEUTROPHIL`.
- **Core Philosophy**: Prioritise technical depth, clear reasoning, and explainable design choices over unnecessary complexity.

---

## Dataset & Preprocessing

- **Source**: Kaggle [Blood Cell Images](https://www.kaggle.com/datasets/paultimothymooney/blood-cells) (`paultimothymooney/blood-cells`).
- **Class Balance**: ~2,480–2,500 images per class. The dataset was checked for class imbalance and found to be well-balanced out of the box, so no class weighting was applied (a deliberate contrast to prior projects where severe imbalance required weighted loss functions).
- **Data Splitting**: Rather than relying on the dataset's default `TRAIN`/`TEST` split, only the pre-existing `TRAIN` folder was used. A custom, stratified **70 / 15 / 15** train/val/test split was performed across all classes to guarantee a proper validation set for hyperparameter tuning and early stopping.

### Environment Fix: Kaggle Path Discovery
During initial data loading in the Kaggle notebook environment, dataset loading failed because newer Kaggle environments nest datasets under `/kaggle/input/datasets/<owner>/<slug>/...` rather than the older flat `/kaggle/input/<slug>/` structure. This was resolved by writing a recursive directory-tree utility script to discover and confirm the exact nested file paths before running any data loading code.

---

## Architecture & The Real Debugging Story

### 1. Initial Attempt & Failure (The Dead-ReLU Problem)
- **Initial Architecture**: Simple CNN (3× `Conv2D` + `MaxPooling2D`, `Flatten`, `Dropout`, `Dense`) trained with the Adam optimizer at `learning_rate=1e-3` and **no** `BatchNormalization`.
- **Symptom**: Training immediately stalled at **~25% accuracy** (random guessing across 4 balanced classes). Both training and validation loss remained completely flat at **~1.386**, which equals $\ln(4)$.
- **Diagnosis**: A classic dead-ReLU / vanishing gradient issue. A learning rate of `1e-3` was too high for a freshly initialized network without activation normalization, causing many ReLU units to output zero early in training and permanently killing gradient flow.

### 2. Fixes & Architectural Decisions
- **Learning Rate & BatchNormalization**: Lowered the learning rate to `1e-4` and inserted `BatchNormalization()` directly after every `Conv2D` layer. This stabilized the distribution of layer activations during training; on the very next run, training loss dropped and accuracy began climbing normally.
- **GlobalAveragePooling2D vs. Flatten**: Replaced `Flatten()` with `GlobalAveragePooling2D()` before the dense classifier.
  - *Reasoning*: With only ~1,700 training images per class, feeding a flattened vector into a dense layer introduces a massive parameter count, significantly increasing overfitting risk. `GlobalAveragePooling2D` spatial-averages each feature map to a single value, dramatically reducing parameter count while matching modern CNN design patterns (e.g., ResNet, EfficientNet).

### 3. Training Dynamics & EarlyStopping
- With the revised architecture, validation accuracy reached **~95% by epoch 14**.
- **Instability Observation**: Beyond epoch 14, occasional sharp validation spikes occurred — e.g., validation accuracy briefly dropped to ~29% for a single epoch before recovering to ~95% in the following epoch.
  - *Analysis*: This sudden, self-correcting drop is distinct from classic overfitting (which manifests as a gradual, widening train/val gap). It was caused by individual large gradient updates temporarily disrupting `BatchNormalization`'s running statistics.
- **Handling**: Managed via `EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)` and `ModelCheckpoint(save_best_only=True)`. Training terminated at epoch 20 (6 epochs after the peak at epoch 14) and successfully restored the exact epoch-14 weights (matching the recorded minimum validation loss of `0.1406`).

---

## Final Results & Error Analysis

Evaluated on the held-out **15% test set** (never seen during training or validation):

- **Overall Test Accuracy**: **96%**

### Classification Report
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **EOSINOPHIL** | 0.88 | 0.98 | 0.93 | 375 |
| **LYMPHOCYTE** | 0.98 | 0.99 | 0.99 | 373 |
| **MONOCYTE** | 0.99 | 1.00 | 0.99 | 372 |
| **NEUTROPHIL** | 0.97 | 0.85 | 0.91 | 375 |
| **Overall / Total** | | | **0.96** | **1495** |

### Confusion Matrix & Error Analysis
- `LYMPHOCYTE` and `MONOCYTE` were classified almost perfectly.
- **Dominant Error**: **49 out of 375 NEUTROPHIL** test images were misclassified as `EOSINOPHIL` — representing the vast majority of errors in the confusion matrix.
- **Explanation**: Both `NEUTROPHIL` and `EOSINOPHIL` are granulocytes with visually similar, multi-lobed nuclei. In clinical practice, they are primarily distinguished by granule staining color (red/pink for eosinophils vs. neutral for neutrophils). The CNN relies more heavily on structural nucleus shape than on fine-grained granule color cues, leading to lower recall (~85%) on neutrophils. This is an explainable limitation of the model and dataset, not a bug.

---

## API & Backend Architecture

The backend is built with **FastAPI** with a strict separation of concerns:

```
app/
├── main.py              # FastAPI app, CORS middleware, and HTTP routes ONLY
├── models/
│   ├── blood_cell_model.keras   # Downloaded/Loaded Keras 3 model
│   └── class_names.json         # ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
├── services/
│   └── inference.py     # Startup downloader, model loading & prediction logic
└── schemas/
    └── prediction.py    # Pydantic request/response schemas
```

### Key Engineering Decisions
1. **Module-Level Model Loading**: The model is loaded **once** at module import time in `services/inference.py`, never per-request, preventing multi-second latency on API calls.
2. **Preprocessing Safety**: The model includes an internal `Rescaling(1./255)` layer. The API converts images to RGB `(224, 224)` float32 arrays in the range `[0, 255]` without manual division to prevent **double-normalization** errors.
3. **Transparent Limitation Reporting**: `POST /predict` returns a `model_note` field in its JSON payload that explicitly surfaces the known `NEUTROPHIL`/`EOSINOPHIL` recall limitation to the caller.
4. **Graceful Startup Degradation**: If the model file is missing or fails to load, `services/inference.py` catches the error. The API server still starts, and `GET /health` reports `model_loaded: false`, allowing deployment health probes to detect the state without crash-looping the server.

---

## Deployment Process & Model Hosting Strategy

### The Challenge: 148 MB Model Binary vs. Git Limits
The trained Keras model (`blood_cell_model.keras`) is **148 MB**, which exceeds GitHub's **100 MB hard limit** for regular file tracking. While Git LFS was initially evaluated, pushing large binary blobs through Git LFS during deployment pipelines can lead to bandwidth bottlenecks, slow clone times, or authentication timeouts in headless build environments.

### The Architectural Solution: Decoupled Startup Fetching
To keep the codebase lightweight (~50 KB) and ensure fast, reliable deployments, the model storage was decoupled from the source repository:

1. **Git Exclusion**: `app/models/*.keras` is ignored in `.gitignore`, keeping binary blobs completely out of the Git history.
2. **GitHub Release Storage**: The `blood_cell_model.keras` binary file was uploaded directly to GitHub Releases under tag **`v2.0`**.
3. **Automated Startup Downloader (`_download_model()`)**:
   - Implemented in `app/services/inference.py` using Python's standard library `urllib.request`.
   - On server startup, the service checks if `app/models/blood_cell_model.keras` is present locally.
   - If missing, it automatically fetches the model from the URL specified by the `MODEL_DOWNLOAD_URL` environment variable.
   - Features a progress reporter that logs progress to stdout every 5 MB and automatically cleans up partial files if the download is interrupted.
4. **Render Blueprint Configuration (`render.yaml`)**:
   - Configured with `runtime: python` and pinned `PYTHON_VERSION: "3.12.0"`.
   - Build Command: `pip install -r requirements.txt` (uses `tensorflow==2.16.1` and `numpy==1.26.4` compatible with Linux `x86_64`).
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - `MODEL_DOWNLOAD_URL`: Points to `https://github.com/Prajjwal-Mahajan/blood-cell-api/releases/download/v2.0/blood_cell_model.keras`.

---

## Setup & Running

### Prerequisites
- Python 3.10–3.12

### 1. Clone Repository
```bash
git clone https://github.com/Prajjwal-Mahajan/blood-cell-api.git
cd blood-cell-api
```

### 2. Virtual Environment & Dependencies
```bash
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Model Setup (Local Running)
Either place `blood_cell_model.keras` inside `app/models/` manually, or set `MODEL_DOWNLOAD_URL`:
```bash
# Optional: Auto-download at startup
export MODEL_DOWNLOAD_URL="https://github.com/Prajjwal-Mahajan/blood-cell-api/releases/download/v2.0/blood_cell_model.keras"
```

### 4. Run Server
```bash
uvicorn app.main:app --reload
```
The server starts at `http://127.0.0.1:8000`.

---

## API Reference

### `GET /`
Returns API metadata and an index of available endpoints.

### `GET /health`
Deployment health check probe.
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /predict`
Accepts a JPEG or PNG image upload via `multipart/form-data`.

**Example Request**:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "accept: application/json" \
     -F "file=@/path/to/cell_image.jpg"
```

**Example Response**:
```json
{
  "predicted_class": "NEUTROPHIL",
  "confidence": 0.7823,
  "model_note": "NOTE: This model has ~85% recall on NEUTROPHIL, most often confused with EOSINOPHIL..."
}
```

**HTTP Status Codes**:
- `200 OK`: Successful prediction.
- `400 Bad Request`: Non-image file type (not JPEG/PNG), empty file upload, or unreadable image bytes.
- `500 Internal Server Error`: Model not loaded or inference error.

---

## Known Limitations

- **Neutrophil vs. Eosinophil Recall**: ~85% recall on `NEUTROPHIL` due to visual similarity of multi-lobed nuclei in granulocytes.
- **Diagnostic Scope**: This is a cell-type identification tool, **not** a disease diagnosis or cell-counting system.
- **Single-Source Data**: Trained on Kaggle blood cell microscopy dataset; generalization to different staining protocols or microscope optics remains unverified.