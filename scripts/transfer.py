#!/usr/bin/env python
"""Create a reviewed source allowlist archive; deployment metadata stays private."""
import argparse
from pathlib import Path
import tarfile
from adaptive_drone_patrol.util import sha256, json_write

ALLOWED_ROOTS = ['src', 'scripts', 'tests', 'configs', 'requirements', 'docs', 'third_party']
ALLOWED_FILES = ['pyproject.toml', 'README.md', 'THIRD_PARTY_NOTICES.md', 'LICENSE', 'spec.md']


def collect_sources():
    files = [Path(f) for f in ALLOWED_FILES if Path(f).is_file()]
    for root in ALLOWED_ROOTS:
        files.extend(f for f in Path(root).rglob('*') if f.is_file() and '__pycache__' not in str(f)
                     and f.suffix not in ['.pyc', '.pt', '.pkl', '.mp4', '.zip']
                     and (not f.name.startswith('execution') or f.name == 'execution.example.yaml'))
    return sorted(set(files))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    output = Path(a.out)
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError(output)
    files = collect_sources()
    with tarfile.open(output, 'w:gz') as archive:
        for f in sorted(set(files)):
            archive.add(f, arcname=str(f))
    json_write(output.with_suffix('.json'), {'files': [{'path': str(f), 'sha256': sha256(f)} for f in files],
                                           'archive_sha256': sha256(output)})
    print(f'Created source allowlist archive: {len(files)} files')


if __name__ == '__main__':
    main()
