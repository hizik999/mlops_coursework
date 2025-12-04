from __future__ import annotations

import pathlib
import re
import string

import pandas as pd

URL_RE = re.compile(r"http\S+|www\.\S+")


def clean_v1(text: str) -> str:
    "Версия v1: простой препроцессинг."
    if not isinstance(text, str):
        return ""
    return text.strip().lower()


def clean_v2(text: str) -> str:
    "Версия v2: более агрессивный препроцессинг."
    text = text.lower().strip()
    text = URL_RE.sub(" ", text)
    text = re.sub(r"\d+", " ", text)
    table = str.maketrans("", "", string.punctuation)
    text = text.translate(table)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_version(
    version: str,
    input_dir: str = "data/splits",
    output_dir: str | None = None,
) -> None:
    cleaner = clean_v1 if version == "v1" else clean_v2

    if output_dir is None:
        output_dir = f"data/processed/{version}"

    input_path = pathlib.Path(input_dir)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        in_file = input_path / f"{split}.csv"
        if not in_file.exists():
            raise FileNotFoundError(f"File not found: {in_file}")

        df = pd.read_csv(in_file)

        if "text" not in df.columns:
            raise KeyError("Column 'text' not found in dataframe")

        df["text"] = df["text"].astype(str).apply(cleaner)
        df.to_csv(output_path / f"{split}.csv", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True, help="v1 or v2")
    parser.add_argument("--input_dir", type=str, default="data/splits")
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()
    preprocess_version(args.version, args.input_dir, args.output_dir)
