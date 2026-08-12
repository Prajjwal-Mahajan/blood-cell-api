from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.schemas.prediction import HealthResponse, PredictionResponse, WelcomeResponse
from app.services import inference

app = FastAPI(
    title="Blood Cell Classification API",
    description=(
        "Classifies white blood cell types from microscopy images using a trained CNN. "
        "**Not a disease diagnosis tool** — results must be reviewed by a medical professional.\n\n"
        "Supported classes: EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=WelcomeResponse, tags=["Info"])
def root() -> WelcomeResponse:
    return WelcomeResponse(
        message="Blood Cell Classification API — visit /docs for the interactive OpenAPI UI.",
        purpose=(
            "Diagnostic-support tool for classifying white blood cell types "
            "(EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL) from microscopy images. "
            "Not a disease diagnosis tool."
        ),
        endpoints={
            "GET  /":        "This welcome message.",
            "GET  /health":  "Health check — confirms model is loaded.",
            "POST /predict": "Upload a JPEG/PNG image; receive cell-type prediction.",
            "GET  /docs":    "Swagger UI.",
            "GET  /redoc":   "ReDoc docs.",
        },
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=inference.is_model_loaded(),
        load_error=inference.get_load_error(),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    responses={
        400: {"description": "Invalid input — wrong file type or empty file."},
        500: {"description": "Inference failed."},
    },
)
async def predict_cell_type(
    file: UploadFile = File(..., description="JPEG or PNG blood cell microscopy image."),
) -> PredictionResponse:
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Please upload a JPEG or PNG image.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        result = inference.predict(image_bytes)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re)) from re
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return PredictionResponse(**result)
