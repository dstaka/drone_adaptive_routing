"""Byte-size/SHA256 recovery receipts; summaries cannot substitute for files."""
import hashlib,json
import os
from pathlib import Path,PurePosixPath


def file_record(path):
    p=Path(path);h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
    return dict(bytes=p.stat().st_size,sha256=h.hexdigest())


def inventory(root):
    root=Path(root)
    if not root.is_dir():raise ValueError('Recovery inventory root is missing or not a directory')
    files={};seen=set();links={}
    def fail(error):raise error
    for directory,dirs,names in os.walk(root,followlinks=True,onerror=fail):
        directory=Path(directory);real=directory.resolve()
        declaration=directory/'declaration.json'
        if declaration.is_file():
            declared=json.loads(declaration.read_text())
            for case in declared.get('cases',[]):
                if not isinstance(case,dict) or 'case_id' not in case:continue
                name=PurePosixPath(case['case_id'])
                if name.is_absolute() or '..' in name.parts:raise ValueError('Unsafe declared case path')
                if not (directory/name).is_dir():raise ValueError('Missing declared recovery case subtree: '+str(name))
        if real in seen:raise ValueError('Cyclic or duplicate symlink subtree in recovery inventory')
        seen.add(real)
        if directory.is_symlink():links[str(directory.relative_to(root))]=str(real)
        for name in dirs+names:
            p=directory/name
            if p.is_symlink() and not p.exists():raise ValueError('Dangling recovery symlink: '+str(p))
        for name in sorted(names):
            p=directory/name
            if not p.is_file():raise ValueError('Unsupported recovery entry: '+str(p))
            files[str(p.relative_to(root))]=file_record(p)
    return dict(schema='elevated-recovery-inventory-v1',directory_symlinks=links,files=files,file_count=len(files),total_bytes=sum(r['bytes'] for r in files.values()))


def verify_inventory(root,expected):
    root=Path(root);actual={}
    for name,record in expected['files'].items():
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe recovery inventory path')
        p=root/name
        if not p.is_file() or file_record(p)!=record:raise ValueError('Recovered file size/SHA mismatch: '+name)
        actual[name]=record
    if len(actual)!=expected['file_count'] or sum(r['bytes'] for r in actual.values())!=expected['total_bytes']:
        raise ValueError('Recovery inventory totals mismatch')
    return dict(all_verified=True,files=len(actual),bytes=expected['total_bytes'])
