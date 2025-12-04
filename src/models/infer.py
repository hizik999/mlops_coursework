import pathlib
from typing import List, Dict

import joblib
import numpy as np
from omegaconf import OmegaConf


class NewsInferenceModel:

    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer

    @classmethod
    def from_config(cls, config_path: str = "configs/inference.yaml") -> "NewsInferenceModel":

        cfg = OmegaConf.load(config_path)
        model_dir = pathlib.Path(cfg.model.path)

        model_path = model_dir / "model.pkl"
        vectorizer_path = model_dir / "vectorizer.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        return cls(model=model, vectorizer=vectorizer)

    def _prepare_texts(self, texts: List[str]) -> List[str]:
        cleaned = []
        for t in texts:
            if t is None:
                cleaned.append("")
            else:
                cleaned.append(str(t))
        return cleaned

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        texts_clean = self._prepare_texts(texts)
        X = self.vectorizer.transform(texts_clean)

        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("Model does not support predict_proba")

        probs = self.model.predict_proba(X)
        return probs

    def predict(self, texts: List[str]) -> List[Dict]:
        """
        Возвращает список словарей:
        {
          "label": int,    # 0 = fake, 1 = real
          "is_fake": bool,
          "score": float   # вероятность класса 1 (real news)
        }
        """
        texts_clean = self._prepare_texts(texts)
        X = self.vectorizer.transform(texts_clean)

        y_pred = self.model.predict(X)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            # probs[:, 1] — вероятность "real" (label=1)
            scores = probs[:, 1]
        else:
            scores = np.full(shape=len(texts_clean), fill_value=np.nan)

        results: List[Dict] = []
        for label, score in zip(y_pred, scores):
            label_int = int(label)
            is_fake = label_int == 0
            score_val = float(score) if not np.isnan(score) else None

            results.append(
                {
                    "label": label_int,
                    "is_fake": is_fake,
                    "score": score_val,
                },
            )

        return results

    def predict_one(self, text: str) -> Dict:
        return self.predict([text])[0]
