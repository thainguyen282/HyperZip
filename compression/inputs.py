"""Read a complete UTF-8 text file without changing its bytes."""
from pathlib import Path


def read_text_file(path):
    """Read one complete UTF-8 file without changing its original bytes."""
    path = Path(path)
    payload = path.read_bytes()
    payload.decode("utf-8")
    return payload
