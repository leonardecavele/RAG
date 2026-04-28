# standard imports
import hashlib
from pathlib import Path


def md5sum(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def file_md5sum(file_path: Path) -> str:
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
