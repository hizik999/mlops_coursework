import pathlib
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_fake_real(raw_dir: str = "data/raw") -> pd.DataFrame:
    raw_path = pathlib.Path(raw_dir)
    fake_path = raw_path / "Fake.csv"
    true_path = raw_path / "True.csv"

    if not fake_path.exists() or not true_path.exists():
        raise FileNotFoundError("Fake.csv or True.csv not found in data/raw")

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        stratify=train_df["label"],
        random_state=random_state,
    )

    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    processed_dir: str = "data/processed/v1",
) -> None:
    out_dir = pathlib.Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default="data/processed/v1")
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()

    df_all = load_raw_fake_real()
    train, val, test = split_dataset(
        df_all,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )
    save_splits(train, val, test, processed_dir=args.processed_dir)
