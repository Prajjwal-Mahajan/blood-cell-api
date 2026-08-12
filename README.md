# Blood Cell Classification API

A production-grade **FastAPI** backend that serves a pre-trained Keras CNN model for classifying white blood cell types from microscopy images.

> **Important disclaimer:** This is a **diagnostic-support tool** for identifying blood cell *type* — it is **not a disease diagnosis tool**. Predictions must always be reviewed by a qualified haematologist or medical professional before any clinical decision is made.

---

## Project Overview

The model classifies microscopy images into one of four white blood cell types:

| Index | Class | Description |
|-------|-------|-------------|
| 0 | **EOSINOPHIL** | Granulocyte; bi-lobed nucleus, red-staining granules |
| 1 | **LYMPHOCYTE** | Agranulocyte; large round nucleus |
| 2 | **MONOCYTE** | Largest WBC; kidney-shaped nucleus |
| 3 | **NEUTROPHIL** | Most common granulocyte; multi-lobed nucleus |

### Known Model Limitation (preserved intentionally)

The model has **lower recall (~85%) on NEUTROPHIL** (test set), most often confused with **EOSINOPHIL**. Both are granulocytes with visually similar multi-lobed nuclei. The feature that more reliably distinguishes them — **granule staining color** — appears to be a subtler cue than nucleus shape, which the model likely under-weights relative to shape. This was identified through confusion matrix analysis on the held-out test set, not assumed in advance.

The API surfaces this limitation in a visible `model_note` field on every prediction where it's relevant — it is **not** silently corrected or hidden.

---

## System Architecture

```
blood-cell-api/
├── app/
│   ├── main.py              <- FastAPI app, routes only, no ML logic
│   ├── models/
│   │   ├── blood_cell_model.keras   <- Trained model (you provide this)
│   │   └── class_names.json         <- ["EOSINOPHIL","LYMPHOCYTE","MONOCYTE","NEUTROPHIL"]
│   ├── services/
│   │   └── inference.py     <- Model load (once at startup) + predict()
│   └── schemas/
│       └── prediction.py    <- Pydantic request/response models
├── requirements.txt
└── README.md
```

### Why is `services/` separated from `main.py`?

**Single Responsibility Principle.** `main.py` handles exactly one concern: HTTP routing (parse request, call service, return response). `services/inference.py` handles exactly one concern: ML inference (load model, preprocess image, run forward pass, decode output).

Benefits of this split:

1. **Readability** — a reviewer can understand the full API surface from `main.py` in ~60 seconds without reading any TensorFlow code.
2. **Testability** — `inference.predict()` can be unit-tested by importing only `services/inference.py`, with no HTTP server involved.
3. **Replaceability** — if you later switch from TensorFlow to ONNX Runtime or PyTorch, you only change `inference.py`. Routes stay untouched.
4. **Interview clarity** — the separation makes the design intent immediately visible.

### Why load the model once at import time?

`tf.keras.models.load_model()` deserialises the full computation graph and loads weights into memory. This takes several seconds. Loading it **per-request** would make every API call multi-second — completely unacceptable.

By assigning the model to a module-level variable in `inference.py`, Python's module system guarantees the code runs **exactly once per process** (subsequent imports reuse the cached module object). The model lives in memory for the lifetime of the server process and is shared safely across all request handlers.

If the model fails to load at startup, the server still starts (rather than crash-looping) and `/health` reports `model_loaded: false`, so a deployment pipeline can detect and alert on the broken state.

---

## ML Pipeline Summary

| Stage | Details |
|---|---|
| Dataset | [Blood Cell Images](https://www.kaggle.com/datasets/paultimothymooney/blood-cells) (Kaggle) — 4 classes, ~2,480–2,500 images/class, already well-balanced |
| Data split | 70/15/15 train/val/test, stratified per class |
| Preprocessing | Resize to 224×224, RGB; model has internal `Rescaling(1./255)` layer |
| Augmentation | Random flip, rotation (±10%), zoom (±10%) — train set only |
| Architecture | 3× (Conv2D → BatchNorm → MaxPool) → GlobalAveragePooling2D → Dropout(0.5) → Dense(128) → Dense(softmax) |
| Class imbalance | Checked, found negligible — no class weighting applied |
| Regularization | Dropout, BatchNorm, EarlyStopping (patience=6, restore best weights) |

### Why GlobalAveragePooling2D instead of Flatten?
With ~1,700 training images per class, `Flatten()` would feed a very large parameter count into the Dense layer, increasing overfitting risk. `GlobalAveragePooling2D` compresses each feature map to a single value first, substantially reducing parameters — this is also the standard choice in modern CNN architectures (ResNet, EfficientNet, etc.), not just Flatten.

### A real debugging story: diagnosing a dead network
Initial training got stuck at ~25% accuracy (random-guess level for 4 classes) with loss flat at `ln(4) ≈ 1.386` across many epochs — the signature of vanishing gradients / dead ReLU units, caused by too high a learning rate (1e-3) on a fresh network with no normalization.

Fixed by:
1. Lowering the learning rate to 1e-4
2. Adding `BatchNormalization()` after each conv layer to keep activations in a stable range

After this fix, the model trained correctly and reached 95%+ validation accuracy within ~14 epochs.

### Training instability, correctly handled
Validation accuracy showed occasional sharp downward spikes during training (e.g., dropping to 29% for a single epoch before recovering) — distinct from classic overfitting (a steady, growing train/val gap), and likely caused by individual large gradient updates temporarily disrupting BatchNorm's running statistics. `EarlyStopping(restore_best_weights=True)` correctly rolled back to the best validation checkpoint rather than keeping the unstable later weights.

---

## Results

**Test set accuracy: 96%**

```
              precision    recall  f1-score   support

  EOSINOPHIL       0.88      0.98      0.93       375
  LYMPHOCYTE       0.98      0.99      0.99       373
    MONOCYTE       0.99      1.00      0.99       372
  NEUTROPHIL       0.97      0.85      0.91       375

    accuracy                           0.96      1495
```

LYMPHOCYTE and MONOCYTE are classified almost perfectly — these cell types have visually distinct nucleus shapes. The dominant error (49 of 375 NEUTROPHIL test images misclassified as EOSINOPHIL) is discussed above under "Known Model Limitation."

---

## Setup

### Prerequisites

- Python 3.10–3.12 (check your installed TensorFlow version's supported Python range before installing)
- The trained model file: `blood_cell_model.keras` — place it at `app/models/blood_cell_model.keras`

### 1 — Create and activate a virtual environment

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Place the model files

```
app/
  models/
    blood_cell_model.keras    <- put your trained model here
    class_names.json          <- already included in this repo
```

---

## Running the Server

```bash
uvicorn app.main:app --reload
```

Server starts on **http://127.0.0.1:8000**.

| URL | Description |
|-----|-------------|
| http://127.0.0.1:8000/ | Welcome message + endpoint index |
| http://127.0.0.1:8000/health | Health check |
| http://127.0.0.1:8000/predict | Prediction endpoint (POST) |
| http://127.0.0.1:8000/docs | Swagger UI (interactive) |
| http://127.0.0.1:8000/redoc | ReDoc documentation |

---

## API Reference

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true
}
```

`model_loaded` is `false` if the model file was missing or failed to load at startup — the server stays up so this remains queryable by a monitoring pipeline.

### `POST /predict`

**Request:** `multipart/form-data`, field `file` (JPEG or PNG).

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "accept: application/json" \
     -F "file=@/path/to/blood_cell_image.jpg"
```

**Success response `200 OK`:**
```json
{
  "predicted_class": "NEUTROPHIL",
  "confidence": 0.7823,
  "model_note": "NOTE: This model has ~85% recall on NEUTROPHIL, most often confused with EOSINOPHIL. Both are granulocytes with visually similar multi-lobed nuclei; the feature that more reliably distinguishes them — granule staining color — appears to be a subtler cue than shape, which the model may under-weight. This is a diagnostic-support tool — not a clinical diagnosis. Always confirm with a qualified haematologist."
}
```

**Error responses:**

| Status | When |
|--------|------|
| `400 Bad Request` | File is not JPEG/PNG, file is empty, or image bytes cannot be decoded |
| `500 Internal Server Error` | Model is not loaded, or an unexpected inference error occurred |

---

## Preprocessing Details

The model has a built-in `Rescaling(1./255)` layer as its first layer. The API does **not** manually divide pixel values by 255 before calling `model.predict()`:

```
Upload bytes
  -> PIL.Image.open() -> .convert("RGB") -> .resize((224, 224))
  -> np.array(dtype=float32)              # values in [0, 255]
  -> np.expand_dims(axis=0)               # add batch dim -> (1, 224, 224, 3)
  -> model.predict()                      # internal Rescaling divides by 255
  -> softmax output shape (4,)
  -> argmax -> class label + confidence
```

Manually normalising beforehand would double-normalise the input and silently produce wrong predictions.

---

## Deployment

Deployed on Render: `[fill in your live URL once deployed]`

---

## Known Limitations

- Lower recall on NEUTROPHIL vs. EOSINOPHIL — see "Known Model Limitation" above.
- Classifies normal cell type only; does not detect disease, abnormal morphology, or count cells.
- Trained on a single public dataset; performance on images from different microscopes, staining protocols, or magnifications is untested.

---

## Dependency Versions

```
tensorflow==2.21.0
fastapi==0.141.1
uvicorn[standard]==0.30.1
python-multipart==0.0.9
pillow==11.3.0
numpy==2.2.0
pydantic==2.13.4
```