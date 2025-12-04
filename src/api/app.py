from fastapi import FastAPI

from src.api.schemas import NewsRequest, NewsResponse

app = FastAPI(
    title="Fake News Classification API",
    description="API для классификации новостей по достоверности",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=NewsResponse)
def predict(request: NewsRequest) -> NewsResponse:
    text = request.text

    # zaglushka
    is_fake = len(text) < 100
    label = 0 if is_fake else 1

    return NewsResponse(
        label=label,
        is_fake=is_fake,
        score=None,
    )
