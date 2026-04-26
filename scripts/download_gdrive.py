#!/usr/bin/env python3
"""
Download dataset from Google Drive shared folder and unzip to data/raw/.

Usage:
    pip install gdown
    python scripts/download_gdrive.py
"""

import os
import zipfile
import tarfile
from pathlib import Path

try:
    import gdown
except ImportError:
    print("Installing gdown...")
    os.system("pip install gdown")
    import gdown


FOLDER_URL = "https://drive.google.com/drive/folders/1_1kwpg3voCcJ3p1G0ZyLpts77A3JvbRS"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def unzip_all(directory: Path):
    """Find and extract all zip/tar/tar.gz files in directory."""
    archives = (
        list(directory.glob("*.zip"))
        + list(directory.glob("*.tar.gz"))
        + list(directory.glob("*.tar"))
        + list(directory.glob("*.tgz"))
    )

    if not archives:
        print("No archives found to extract.")
        return

    for archive in sorted(archives):
        print(f"\nExtracting: {archive.name}")
        extract_dir = directory / archive.stem.replace(".tar", "")

        try:
            if archive.suffix == ".zip":
                with zipfile.ZipFile(archive, "r") as zf:
                    zf.extractall(extract_dir)
            elif archive.suffix in (".gz", ".tgz") or ".tar" in archive.name:
                with tarfile.open(archive, "r:*") as tf:
                    tf.extractall(extract_dir)

            print(f"  → Extracted to: {extract_dir.name}/")

            # Count extracted files
            count = sum(1 for _ in extract_dir.rglob("*") if _.is_file())
            print(f"  → {count} files extracted")
        except Exception as e:
            print(f"  ✗ Failed to extract {archive.name}: {e}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download
    print(f"Downloading from: {FOLDER_URL}")
    print(f"Saving to: {OUTPUT_DIR}\n")
    gdown.download_folder(url=FOLDER_URL, output=str(OUTPUT_DIR), quiet=False)

    # Step 2: Unzip all archives
    print("\n" + "=" * 60)
    print("Extracting archives...")
    print("=" * 60)
    unzip_all(OUTPUT_DIR)

    # Step 3: Summary
    print("\n" + "=" * 60)
    print("Done! Final contents of data/raw/:")
    print("=" * 60)
    for item in sorted(OUTPUT_DIR.iterdir()):
        if item.is_dir():
            count = sum(1 for _ in item.rglob("*") if _.is_file())
            print(f"  📁 {item.name}/  ({count} files)")
        else:
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  📦 {item.name}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
