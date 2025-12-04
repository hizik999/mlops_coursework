from __future__ import annotations
from fastapi import FastAPI, HTTPException

from src.api.schemas import NewsRequest
from src.models.infer import NewsInferenceModel

app = FastAPI(
    title="Fake News Classification API",
    description="API для классификации новостей по достоверности (fake / real)",
    version="0.2.0",
)

model: NewsInferenceModel | None = None


@app.on_event("startup")
def load_model() -> None:
    global model
    try:
        model = NewsInferenceModel.from_config("configs/inference.yaml")
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load model: {exc}")
        model = None


@app.get("/health")
def health() -> dict:
    status = "200 OK" if model is not None else "model_not_loaded"
    return {"status": status}


@app.post("/predict")
def predict(request: NewsRequest):
    model = getattr(app.state, "model", None)

    if model is None:
        # модель не загружена – значит сервис недоступен
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Cannot perform prediction.",
        )
    try:
        result = model.predict([request.text])[0]
        return result
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
