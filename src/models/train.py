import json
import pathlib
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.features.build_features import build_tfidf_features


def load_splits(
    processed_base_dir: str, version: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = pathlib.Path(processed_base_dir) / version
    train_df = pd.read_csv(base / "train.csv")
    val_df = pd.read_csv(base / "val.csv")
    test_df = pd.read_csv(base / "test.csv")
    return train_df, val_df, test_df


def _sanitize_text_columns(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for df_name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        if "text" not in df.columns:
            raise KeyError(f"Column 'text' not found in {df_name} dataframe")

        df["text"] = df["text"].fillna("").astype(str)

    return train_df, val_df, test_df


def compute_metrics(
    y_val_true: np.ndarray,
    y_val_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
) -> Dict[str, float]:
    metrics = {
        "val_accuracy": float(accuracy_score(y_val_true, y_val_pred)),
        "val_f1": float(f1_score(y_val_true, y_val_pred)),
        "test_accuracy": float(accuracy_score(y_test_true, y_test_pred)),
        "test_f1": float(f1_score(y_test_true, y_test_pred)),
    }
    return metrics


def save_metrics_and_models(
    cfg,
    model,
    vectorizer,
    metrics: Dict[str, float],
) -> None:
    models_dir = pathlib.Path(cfg.output.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    run_dir = models_dir / cfg.output.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, run_dir / "model.pkl")
    joblib.dump(vectorizer, run_dir / "vectorizer.pkl")

    run_metrics_path = run_dir / "metrics.json"
    with run_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    root_metrics_path = pathlib.Path(cfg.output.metrics_path)
    root_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with root_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def train_logreg_tfidf(cfg) -> Dict[str, float]:
    train_df, val_df, test_df = load_splits(
        cfg.data.base_dir,
        cfg.data.version,
    )

    train_df, val_df, test_df = _sanitize_text_columns(train_df, val_df, test_df)

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

    metrics = compute_metrics(y_val, y_val_pred, y_test, y_test_pred)
    save_metrics_and_models(cfg, model, vectorizer, metrics)

    return metrics


def train_rf_tfidf(cfg) -> Dict[str, float]:
    """
    Обучение RandomForestClassifier + TF-IDF.
    """
    train_df, val_df, test_df = load_splits(
        cfg.data.base_dir,
        cfg.data.version,
    )

    train_df, val_df, test_df = _sanitize_text_columns(train_df, val_df, test_df)

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

    model = RandomForestClassifier(
        n_estimators=cfg.model.rf.n_estimators,
        max_depth=cfg.model.rf.max_depth,
        n_jobs=-1,
        random_state=cfg.train.random_state,
    )
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    metrics = compute_metrics(y_val, y_val_pred, y_test, y_test_pred)
    save_metrics_and_models(cfg, model, vectorizer, metrics)

    return metrics


def main(config_path: str) -> None:
    cfg = OmegaConf.load(config_path)

    if cfg.model.name == "logreg":
        metrics = train_logreg_tfidf(cfg)
    elif cfg.model.name == "rf":
        metrics = train_rf_tfidf(cfg)
    else:
        raise NotImplementedError(
            f"Model '{cfg.model.name}' is not implemented yet in train.py",
        )

    print("Training finished. Metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


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
    main(args.config)
