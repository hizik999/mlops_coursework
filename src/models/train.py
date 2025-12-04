import json
import pathlib

import joblib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.features.build_features import build_tfidf_features


def load_splits(processed_base_dir: str, version: str):
    base = pathlib.Path(processed_base_dir) / version
    train_df = pd.read_csv(base / "train.csv")
    val_df = pd.read_csv(base / "val.csv")
    test_df = pd.read_csv(base / "test.csv")
    return train_df, val_df, test_df


def train_logreg_tfidf(cfg) -> dict:
    train_df, val_df, test_df = load_splits(
        cfg.data.base_dir,
        cfg.data.version,
    )

    X_train, X_val, X_test, vectorizer = build_tfidf_features(
        train_df["text"],
        val_df["text"],
        test_df["text"],
        max_features=cfg.features.tfidf.max_features,
        ngram_range=tuple(cfg.features.tfidf.ngram_range),
        min_df=cfg.features.tfidf.min_df,
    )

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    model = LogisticRegression(
        C=cfg.model.logreg.C,
        max_iter=cfg.model.logreg.max_iter,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    metrics = {
        "val_accuracy": float(accuracy_score(y_val, y_val_pred)),
        "val_f1": float(f1_score(y_val, y_val_pred)),
        "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
        "test_f1": float(f1_score(y_test, y_test_pred)),
    }

    models_dir = pathlib.Path(cfg.output.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    run_dir = models_dir / cfg.output.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, run_dir / "model.pkl")
    joblib.dump(vectorizer, run_dir / "vectorizer.pkl")

    with open(run_dir / cfg.output.metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to train config",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    if cfg.model.name == "logreg":
        metrics = train_logreg_tfidf(cfg)
    else:
        raise NotImplementedError(
            f"Model '{cfg.model.name}' is not implemented yet in train.py",
        )

    print("Training finished. Metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
