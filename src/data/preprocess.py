import pathlib

import pandas as pd


def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.strip().lower()


def preprocess_version(
    version: str,
    input_dir: str = "data/processed/v1",
    output_dir: str | None = None,
) -> None:
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

        df["text"] = df["text"].astype(str).apply(basic_clean)
        df.to_csv(output_path / f"{split}.csv", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True, help="v1 or v2")
    parser.add_argument("--input_dir", type=str, default="data/processed/v1")
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()
    preprocess_version(args.version, args.input_dir, args.output_dir)
