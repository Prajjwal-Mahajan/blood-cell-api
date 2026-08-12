from pydantic import BaseModel, ConfigDict, Field


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predicted_class: str = Field(..., examples=["NEUTROPHIL"])
    confidence: float = Field(..., ge=0.0, le=1.0, examples=[0.9213])
    model_note: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., examples=["ok"])
    model_loaded: bool


class WelcomeResponse(BaseModel):
    message: str
    purpose: str
    endpoints: dict
