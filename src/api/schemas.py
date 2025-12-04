from pydantic import BaseModel


class NewsRequest(BaseModel):
    text: str


class NewsResponse(BaseModel):
    label: int
    is_fake: bool
    score: float | None = None
