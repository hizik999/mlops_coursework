from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_features(
    train_texts: pd.Series,
    val_texts: pd.Series | None = None,
    test_texts: pd.Series | None = None,
    max_features: int = 50000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
) -> Tuple:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
    )

    X_train = vectorizer.fit_transform(train_texts)

    X_val = vectorizer.transform(val_texts) if val_texts is not None else None
    X_test = vectorizer.transform(test_texts) if test_texts is not None else None

    return X_train, X_val, X_test, vectorizer
