import json
import os
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = _MODELS_DIR / "blood_cell_model.keras"
CLASS_NAMES_PATH = _MODELS_DIR / "class_names.json"

IMG_SIZE: tuple[int, int] = (224, 224)

_NEUTROPHIL_NOTE = (
    "NOTE: This model has ~85% recall on NEUTROPHIL, most often confused "
    "with EOSINOPHIL. Both are granulocytes with visually similar "
    "multi-lobed nuclei; the feature that more reliably distinguishes them "
    "— granule staining color — appears to be a subtler cue than shape, "
    "which the model may under-weight. This is a diagnostic-support tool "
    "— not a clinical diagnosis. Always confirm with a qualified haematologist."
)
_GENERAL_NOTE = (
    "This is a diagnostic-support tool for blood cell type classification, "
    "not a disease diagnosis tool. Results should be reviewed by a qualified "
    "medical professional."
)


def _download_model() -> None:
    """Download the model from MODEL_DOWNLOAD_URL if not already on disk."""
    if MODEL_PATH.exists():
        return

    url = os.environ.get("MODEL_DOWNLOAD_URL", "").strip()
    if not url:
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH} and MODEL_DOWNLOAD_URL is not set."
        )

    print(f"[inference] Downloading model from {url}")
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    _last: list[int] = [0]

    def _progress(blocks: int, block_size: int, total: int) -> None:
        mb = blocks * block_size / (1024 * 1024)
        if mb - _last[0] >= 5:
            _last[0] = int(mb)
            print(f"[inference]   {mb:.1f} MB / {total / (1024*1024):.1f} MB")

    try:
        urllib.request.urlretrieve(url, MODEL_PATH, reporthook=_progress)
        print(f"[inference] Download complete.")
    except Exception as exc:
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        raise RuntimeError(f"Failed to download model: {exc}") from exc


# Load once at startup — reused across all requests
_model_loaded: bool = False
model: tf.keras.Model | None = None
class_names: list[str] = []

try:
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"class_names.json not found at {CLASS_NAMES_PATH}.")

    _download_model()

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)

    _model_loaded = True
    print("[inference] Model loaded successfully.")

except Exception as _load_error:
    print(f"[inference] WARNING — model failed to load: {_load_error}")
    _model_loaded = False


def is_model_loaded() -> bool:
    return _model_loaded


def predict(image_bytes: bytes) -> dict:
    if not _model_loaded or model is None:
        raise RuntimeError("Model is not loaded. Check server logs.")

    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    except Exception as e:
        raise ValueError(f"Cannot decode image: {e}") from e

    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    # Do NOT divide by 255 — the model has a built-in Rescaling(1./255) layer.

    preds: np.ndarray = model.predict(arr, verbose=0)[0]

    predicted_idx = int(np.argmax(preds))
    predicted_class = class_names[predicted_idx]
    confidence = float(preds[predicted_idx])

    note = _NEUTROPHIL_NOTE if predicted_class in ("NEUTROPHIL", "EOSINOPHIL") else _GENERAL_NOTE

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "model_note": note,
    }
