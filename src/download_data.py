"""Download Steam dataset from KaggleHub and optionally copy into local data/."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

import kagglehub


DATASET_HANDLE = "antonkozyriev/game-recommendations-on-steam"
EXPECTED_FILES = ("recommendations.csv", "games.csv", "users.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Steam recommendation dataset via KaggleHub."
    )
    parser.add_argument(
        "--copy-to-data",
        action="store_true",
        help="Copy required CSVs into local ./data after download.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Destination data directory used with --copy-to-data.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in destination directory.",
    )
    return parser.parse_args()


def validate_dataset_files(dataset_path: Path, expected_files: Iterable[str]) -> None:
    missing = [name for name in expected_files if not (dataset_path / name).exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            f"Downloaded path is missing required files: {missing_str}. "
            f"Found path: {dataset_path}"
        )


def copy_dataset_files(dataset_path: Path, destination: Path, overwrite: bool = False) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for filename in EXPECTED_FILES:
        src = dataset_path / filename
        dst = destination / filename
        if dst.exists() and not overwrite:
            print(f"Skipping existing file: {dst}")
            continue
        shutil.copy2(src, dst)
        print(f"Copied: {src} -> {dst}")


def main() -> None:
    args = parse_args()

    downloaded_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
    print(f"KaggleHub dataset path: {downloaded_path}")

    validate_dataset_files(downloaded_path, EXPECTED_FILES)
    print("Found required files: recommendations.csv, games.csv, users.csv")

    if args.copy_to_data:
        copy_dataset_files(downloaded_path, args.data_dir, overwrite=args.overwrite)
        print(f"Local data directory ready at: {args.data_dir.resolve()}")


if __name__ == "__main__":
    main()

