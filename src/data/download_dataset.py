import pathlib
import shutil


def download_dataset(source_path: str, target_dir: str = "data/raw") -> None:
    target_path = pathlib.Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    src = pathlib.Path(source_path)

    if src.is_dir():
        for item in src.iterdir():
            if item.is_file():
                shutil.copy(item, target_path / item.name)
    elif src.is_file():
        shutil.copy(src, target_path / src.name)
    else:
        raise FileNotFoundError(f"Source path not found: {source_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_path", type=str, help="Path to Kaggle dataset (dir or file)"
    )
    parser.add_argument("--target_dir", type=str, default="data/raw")

    args = parser.parse_args()
    download_dataset(args.source_path, args.target_dir)
