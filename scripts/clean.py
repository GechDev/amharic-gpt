#!/usr/bin/env python3
"""
Clean script to remove caches, temporary files, and build artifacts.
Cross-platform (Windows, Linux, macOS).
"""

import os
import shutil
import sys

TARGET_DIRS = [
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
    "htmlcov",
]

TARGET_EXTS = [
    ".pyc",
    ".pyo",
    ".pyd",
]

TARGET_FILES = [
    ".coverage",
]


def clean_project(root_dir: str = "."):
    deleted_dirs = 0
    deleted_files = 0

    for root, dirs, files in os.walk(root_dir, topdown=False):
        # Skip git directory
        if ".git" in root.split(os.sep):
            continue

        for d in dirs:
            if d in TARGET_DIRS:
                full_path = os.path.join(root, d)
                try:
                    shutil.rmtree(full_path)
                    print(f"Removed directory: {full_path}")
                    deleted_dirs += 1
                except Exception as e:
                    print(f"Failed to remove directory {full_path}: {e}")

        for f in files:
            full_path = os.path.join(root, f)
            if any(f.endswith(ext) for ext in TARGET_EXTS) or f in TARGET_FILES:
                try:
                    os.remove(full_path)
                    print(f"Removed file: {full_path}")
                    deleted_files += 1
                except Exception as e:
                    print(f"Failed to remove file {full_path}: {e}")

    print(f"\n[Clean Summary] Deleted {deleted_dirs} directories and {deleted_files} files.")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_project(base_dir)
