"""Strict local recovery and supplied human-review receipts for dual artifacts.

Inventory and recovery require closed writers. No filesystem walk can provide a
transactional snapshot of a concurrently changing tree. Every file is checked
for change while hashed; recovery independently inventories before and after
copying, including a new destination inventory. Source files are never removed.

An inventory is an external document: save it outside the inventoried tree, or
explicitly exclude its exact relative filename. Exclusions never match globs or
whole subtrees. Recovery returns its receipt without writing it into either tree.
This module performs no remote transfer and never creates a human approval.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat

INVENTORY_SCHEMA = 'dual-artifact-inventory-v1'
MEDIA_ROLES = frozenset(('normal','priority-transfer','crossing-yield'))
SHA256 = re.compile(r'[0-9a-f]{64}\Z')


def _relative(name):
    if (not isinstance(name,str) or not name or '\\' in name or '\x00' in name
            or any(part in ('','.','..') or ':' in part for part in name.split('/'))
            or PurePosixPath(name).is_absolute()):
        raise ValueError(f'Unsafe relative artifact path: {name!r}')
    return name


def _root(root, *, must_exist=True):
    path = Path(root).absolute()
    # Do not silently follow a user-provided symlink to a different artifact set.
    for part in (path,*path.parents):
        if part.is_symlink():
            raise ValueError(f'Symlink artifact path is forbidden: {part}')
    if must_exist and not path.is_dir():
        raise ValueError('Artifact root is missing or not a directory')
    return path.resolve(strict=must_exist)


def _exclusions(exclude):
    if isinstance(exclude,(str,bytes)):
        raise ValueError('Exclusions must be a collection of exact relative filenames')
    result = [_relative(name) for name in exclude]
    if len(set(result)) != len(result) or any(any(char in name for char in '*?[]') for name in result):
        raise ValueError('Exclusions must be unique explicit filenames, not patterns')
    return sorted(result)


def _record_valid(record):
    if (not isinstance(record,dict) or set(record) != {'bytes','sha256'}
            or type(record['bytes']) is not int or record['bytes'] < 0
            or not isinstance(record['sha256'],str) or SHA256.fullmatch(record['sha256']) is None):
        raise ValueError('Artifact record requires a nonnegative integer byte count and SHA256')


@contextmanager
def _open_regular(root, name):
    """Open every component without following links, including during copies."""
    name = _relative(name)
    directory = os.open(root,os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parts = name.split('/')
        for part in parts[:-1]:
            child = os.open(part,os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(parts[-1],os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,dir_fd=directory)
        with os.fdopen(descriptor,'rb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError(f'Only regular artifact files are allowed: {name}')
            yield stream,directory,parts[-1]
    except OSError as error:
        raise ValueError(f'Artifact is missing, inaccessible, or linked: {name}') from error
    finally:
        os.close(directory)


def _file_record(root, name):
    """Hash real bytes and reject changes to content or identity during reading."""
    with _open_regular(root,name) as (stream,directory,basename):
        before = os.fstat(stream.fileno())
        digest = hashlib.sha256()
        size = 0
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(stream.fileno())
        current = os.stat(basename,dir_fd=directory,follow_symlinks=False)

        def fingerprint(s):
            return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

        if fingerprint(before) != fingerprint(after) or fingerprint(after) != fingerprint(current) or size != after.st_size:
            raise ValueError(f'Artifact changed while hashing: {name}')
        return dict(bytes=size,sha256=digest.hexdigest())


def inventory(root, exclude=()):
    """Sorted relative path/byte/SHA256 inventory; symlinks and special files fail."""
    root = _root(root)
    excluded = _exclusions(exclude)
    files = {}

    def fail(error):
        raise error

    for directory,dirs,names in os.walk(root,followlinks=False,onerror=fail):
        directory = Path(directory)
        for name in sorted(dirs+names):
            path = directory/name
            relative = path.relative_to(root).as_posix()
            _relative(relative)
            if path.is_symlink():
                raise ValueError(f'Symlink artifact is forbidden: {relative}')
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                if relative in excluded:
                    raise ValueError('Exclusions cannot hide directory subtrees')
            elif not stat.S_ISREG(mode):
                raise ValueError(f'Only regular files and directories are supported: {relative}')
            elif relative not in excluded:
                files[relative] = _file_record(root,relative)
    files = dict(sorted(files.items()))
    return dict(schema=INVENTORY_SCHEMA,files=files,file_count=len(files),
                total_bytes=sum(record['bytes'] for record in files.values()),excluded=excluded)


def _validate_inventory(value):
    required = {'schema','files','file_count','total_bytes','excluded'}
    if not isinstance(value,dict) or set(value) != required or value['schema'] != INVENTORY_SCHEMA:
        raise ValueError('Exact dual-artifact-inventory-v1 manifest required')
    if not isinstance(value['files'],dict) or not isinstance(value['excluded'],list):
        raise ValueError('Invalid artifact file or exclusion collection')
    excluded = _exclusions(value['excluded'])
    names = value['files']
    for name,record in names.items():
        _relative(name)
        _record_valid(record)
        if name in excluded or any(parent.as_posix() in names or parent.as_posix() in excluded
                                   for parent in PurePosixPath(name).parents if str(parent) != '.'):
            raise ValueError('Inventory contains an excluded path or a file/directory collision')
    if (type(value['file_count']) is not int or value['file_count'] != len(names)
            or type(value['total_bytes']) is not int
            or value['total_bytes'] != sum(record['bytes'] for record in names.values())):
        raise ValueError('Inventory totals do not match declared file records')
    return value


def compare_inventories(source, destination):
    """Fail on malformed manifests, missing/extra paths, hashes, sizes or exclusions."""
    source = _validate_inventory(source)
    destination = _validate_inventory(destination)
    expected,actual = source['files'],destination['files']
    missing = sorted(set(expected)-set(actual))
    extra = sorted(set(actual)-set(expected))
    mismatch = sorted(name for name in set(expected)&set(actual) if expected[name] != actual[name])
    if missing or extra or mismatch or source['excluded'] != destination['excluded']:
        raise ValueError(f'Inventory mismatch: missing={missing}; extra={extra}; mismatch={mismatch}; '
                         f'exclusions_match={source["excluded"] == destination["excluded"]}')
    return dict(verified=True,file_count=source['file_count'],total_bytes=source['total_bytes'])


def recover_local(source, destination, exclude=()):
    """Copy to a new local directory, retaining source and any failed partial copy.

    Callers must close writers first. An existing destination is never changed.
    A failed partial destination is retained for diagnosis; retry with a new path.
    Inventory documents are returned for the caller to save outside both trees.
    """
    source = _root(source)
    destination = _root(destination,must_exist=False)
    if destination.exists() or destination == source or source in destination.parents:
        raise ValueError('Recovery requires a new destination outside the source tree')
    if not destination.parent.is_dir():
        raise ValueError('Recovery destination parent directory must already exist')
    exclude = tuple(_exclusions(exclude))
    before = inventory(source,exclude)
    destination.mkdir()
    for name in before['files']:
        # Secure source opens never follow links. Independent before/after
        # inventories reject changes to the source or the recovered byte set.
        if _file_record(source,name) != before['files'][name]:
            raise ValueError(f'Source artifact changed before copy: {name}')
        target = destination/name
        target.parent.mkdir(parents=True,exist_ok=True)
        with _open_regular(source,name) as (origin,_,_), target.open('xb') as output:
            shutil.copyfileobj(origin,output,1024*1024)
            output.flush()
            os.fsync(output.fileno())
    after = inventory(source,exclude)
    recovered = inventory(destination,exclude)
    compare_inventories(before,after)
    comparison = compare_inventories(after,recovered)
    return dict(schema='dual-local-recovery-v1',transport='local-filesystem-copy',
                remote_transfer_verified=False,source=str(source),destination=str(destination),
                source_retained=True,source_inventory=after,destination_inventory=recovered,
                verified_at=datetime.now(timezone.utc).isoformat(),**comparison)


def _timestamp(value):
    if not isinstance(value,str):
        raise ValueError('Timezone-aware review timestamp required')
    try:
        stamp = datetime.fromisoformat(value.replace('Z','+00:00'))
    except ValueError as error:
        raise ValueError('Invalid review timestamp') from error
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError('Timezone-aware review timestamp required')
    return stamp


def verify_human_reviews(media_manifest, receipts, *, media_root):
    """Validate supplied human attestations against three current final file hashes.

    Manifest: dual-media-v1 with ``finalized_at`` and three ``videos`` records,
    each containing role, relative path, bytes and sha256. Roles are normal,
    priority-transfer and crossing-yield. Additional machine metrics are allowed.
    Receipts must identify path, sha256, reviewer, reviewed_at, full_length=True,
    playback_speed=1 and approved=True. Review time cannot precede finalization
    or lie in the future according to the verifier clock.

    This validates the supplied attestation; it does not independently establish
    the reviewer's identity or fabricate approval. Encoding, duration, projected
    visibility, safety and content checks remain separate mandatory media gates.
    """
    if (not isinstance(media_manifest,dict) or media_manifest.get('schema') != 'dual-media-v1'
            or not isinstance(media_manifest.get('videos'),list) or len(media_manifest['videos']) != 3):
        raise ValueError('Exactly three final dual-media-v1 videos required')
    finalized = _timestamp(media_manifest.get('finalized_at'))
    verified_at = datetime.now(timezone.utc)
    if finalized > verified_at:
        raise ValueError('Media finalization cannot be in the future')
    root = _root(media_root)
    videos = {}
    roles = set()
    for video in media_manifest['videos']:
        if not isinstance(video,dict) or not {'role','path','bytes','sha256'} <= set(video):
            raise ValueError('Final video path, role, size and SHA256 required')
        name = _relative(video['path'])
        _record_valid(dict(bytes=video['bytes'],sha256=video['sha256']))
        if (video['bytes'] == 0 or name in videos or not isinstance(video['role'],str)
                or video['role'] not in MEDIA_ROLES or video['role'] in roles):
            raise ValueError('Three distinct nonempty final video paths and scenario roles required')
        videos[name] = video
        roles.add(video['role'])
        if _file_record(root,name) != dict(bytes=video['bytes'],sha256=video['sha256']):
            raise ValueError(f'Actual final media size/SHA256 mismatch: {name}')
    if not isinstance(receipts,list) or len(receipts) != 3:
        raise ValueError('Exactly three supplied human-review receipts required')
    seen = set()
    for receipt in receipts:
        required = {'path','sha256','reviewer','reviewed_at','full_length','playback_speed','approved'}
        if not isinstance(receipt,dict) or not required <= set(receipt):
            raise ValueError('Incomplete human-review receipt')
        name = _relative(receipt['path'])
        if name not in videos or name in seen:
            raise ValueError('Human reviews must cover each exact final file once')
        seen.add(name)
        if (receipt['sha256'] != videos[name]['sha256']
                or receipt['full_length'] is not True or receipt['approved'] is not True
                or type(receipt['playback_speed']) not in (int,float) or receipt['playback_speed'] != 1
                or not isinstance(receipt['reviewer'],str) or not receipt['reviewer'].strip()
                or not finalized <= _timestamp(receipt['reviewed_at']) <= verified_at):
            raise ValueError('Review is stale, hash-mismatched, partial, non-1x, or lacks human approval')
    return dict(verified=True,reviewed_files=3,actual_file_integrity_verified=True,
                machine_verified_media=False,reviewer_identity_independently_verified=False,
                final_sha256={name:video['sha256'] for name,video in sorted(videos.items())})
