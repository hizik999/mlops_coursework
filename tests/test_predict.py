from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.models.infer import NewsInferenceModel

app = FastAPI()

model: NewsInferenceModel | None = None


class NewsRequest(BaseModel):
    text: str


@app.on_event("startup")
def load_model():
    global model
    try:
        model = NewsInferenceModel.from_paths(
            model_path="models/best_model/model.pkl",
            vectorizer_path="models/best_model/vectorizer.pkl",
        )
    except Exception:
        model = None  # модель не загрузилась


@app.get("/health")
def health():
    return {"status": "ok" if model else "model_not_loaded"}


@app.post("/predict")
def predict(request: NewsRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Cannot perform prediction.")

    try:
        result = model.predict([request.text])
        return {"result": result[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
