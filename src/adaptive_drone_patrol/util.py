"""Portable file integrity, configuration and output helpers."""

import hashlib
import json
from pathlib import Path
import yaml


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_config(path):
    with Path(path).open() as stream:
        result = yaml.safe_load(stream)
    if not isinstance(result, dict):
        raise ValueError(f'Expected mapping in {path}')
    return result


def json_write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def unique_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path
