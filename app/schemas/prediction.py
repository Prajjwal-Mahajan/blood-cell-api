from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    predicted_class: str = Field(..., examples=["NEUTROPHIL"])
    confidence: float = Field(..., ge=0.0, le=1.0, examples=[0.9213])
    model_note: str


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    model_loaded: bool


class WelcomeResponse(BaseModel):
    message: str
    purpose: str
    endpoints: dict
