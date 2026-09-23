"""Whole-file UTF-8 input and shared path validation."""
from pathlib import Path


def contained_path(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    path = (root / relative).resolve()
    if relative.is_absolute() or path == root or root not in path.parents:
        raise ValueError(f"Path escapes input/output directory: {relative}")
    return path


def read_text_file(path):
    """Read one complete UTF-8 file without changing its original bytes."""
    path = Path(path)
    payload = path.read_bytes()
    payload.decode("utf-8")
    return payload
