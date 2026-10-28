"""Lossless, indexed dual-native trace storage (storage format version 1).

The payload remains the exact JSON data model: ordered string-key dictionaries,
lists, finite Python floats, arbitrary Python integers, strings, booleans, null.
Homogeneous numeric arrays and scalar numbers use little-endian IEEE float64 /
int64 pools; larger integers remain decimal strings. No downcasting, quantizing,
delta rounding, stage removal, or geometry removal occurs. Pools are byte-shuffled
and zlib compressed with structural metadata in independently hashed chunks.

Supports direct dual-raw-v1 records with world_time and consistently optional
tick, or native probe RESET/STEP wrappers with actual.schema/actual.world_time.
RESET's missing tick is treated as zero for validation only. Dictionary contents
are never augmented. A mandatory indexed SHA-256 footer rejects incomplete files;
read_raw validates every chunk and record before yielding the first record.

Checksums detect corruption, not adversarial replacement of the whole artifact.
Bind the archive's external SHA-256 to the campaign inventory. Never delete a
source trace until independent decoded equality and artifact verification pass.
"""
from __future__ import annotations

from array import array
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import struct
import zlib

import numpy as np

_MAGIC = b'DUALRAW\x00\x01'
_U64 = struct.Struct('<Q')
_HEADER = struct.Struct('<QQQ')
_MAX_BLOCK = 1 << 31
_IDENTITIES = ('case_id', 'model_hash', 'policy_hash', 'profile_hash')


def _numeric(value):
    if not isinstance(value, list) or not value:
        return None
    first = value[0]
    if type(first) in (int, float):
        typ = type(first)
        if not all(type(v) is typ for v in value):
            return None
        if typ is int and not all(-(1 << 63) <= v < (1 << 63) for v in value):
            return None
        if typ is float and not all(math.isfinite(v) for v in value):
            raise ValueError('Nonfinite numbers cannot enter strict native JSON raw')
        return ('i' if typ is int else 'f'), [len(value)], value
    if isinstance(first, list):
        parts = [_numeric(v) for v in value]
        if any(p is None for p in parts) or any(p[:2] != parts[0][:2] for p in parts):
            return None
        return parts[0][0], [len(value)]+parts[0][1], [v for p in parts for v in p[2]]
    return None


def _encode(value, floats, integers):
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('Nonfinite numbers cannot enter strict native JSON raw')
        index = len(floats)
        floats.append(value)
        return ['f', index]
    if type(value) is int:
        if not -(1 << 63) <= value < (1 << 63):
            return ['z', str(value)]
        index = len(integers)
        integers.append(value)
        return ['i', index]
    if value is None or type(value) in (str, bool):
        return ['v', value]
    if type(value) is dict:
        if not all(type(k) is str for k in value):
            raise ValueError('Raw JSON dictionary keys must be strings')
        return ['d', [[k, _encode(v, floats, integers)] for k, v in value.items()]]
    if type(value) is list:
        numeric = _numeric(value)
        if numeric is not None:
            dtype, shape, flat = numeric
            pool = floats if dtype == 'f' else integers
            start = len(pool)
            pool.extend(flat)
            return ['a', dtype, shape, start, len(flat)]
        # RK4 stages and substeps repeat the same fields. Store their columns,
        # turning lists of qpos/geom matrices into large binary arrays.
        if value and all(type(v) is dict for v in value):
            keys = list(value[0])
            if all(list(v) == keys for v in value):
                if not all(type(k) is str for k in keys):
                    raise ValueError('Raw JSON dictionary keys must be strings')
                return ['c', len(value), [[k, _encode([v[k] for v in value], floats, integers)] for k in keys]]
        return ['l', [_encode(v, floats, integers) for v in value]]
    raise ValueError(f'Unsupported raw JSON value type: {type(value).__name__}')


def _decode(node, floats, integers):
    tag = node[0]
    if tag == 'f':
        return float(floats[node[1]])
    if tag == 'i':
        return int(integers[node[1]])
    if tag == 'z':
        return int(node[1])
    if tag == 'v':
        return node[1]
    if tag == 'd':
        return {k: _decode(v, floats, integers) for k, v in node[1]}
    if tag == 'l':
        return [_decode(v, floats, integers) for v in node[1]]
    if tag == 'a':
        _, dtype, shape, start, count = node
        pool = floats if dtype == 'f' else integers
        return pool[start:start+count].reshape(shape).tolist()
    if tag == 'c':
        columns = [(k, _decode(v, floats, integers)) for k, v in node[2]]
        return [{k: column[i] for k, column in columns} for i in range(node[1])]
    raise ValueError('Unknown typed raw node')


def _shuffle(pool, dtype):
    data = np.asarray(pool, dtype=dtype)
    return data.view(np.uint8).reshape(-1, 8).T.tobytes()


def _unshuffle(data, dtype):
    return np.frombuffer(np.frombuffer(data, dtype=np.uint8).reshape(8, -1).T.copy().tobytes(), dtype=dtype)


def _pack(trees, floats, integers, level):
    metadata = json.dumps(trees, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    payload = _HEADER.pack(len(metadata), len(floats), len(integers))+metadata
    payload += _shuffle(floats, '<f8')+_shuffle(integers, '<i8')
    return zlib.compress(payload, level=level)


def _unpack(payload):
    try:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(payload)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail or len(raw) < _HEADER.size:
            raise ValueError('Incomplete or trailing compressed chunk')
        size, nf, ni = _HEADER.unpack_from(raw)
        if len(raw) != _HEADER.size+size+8*(nf+ni):
            raise ValueError('Numeric pool length mismatch')
        start = _HEADER.size+size
        trees = json.loads(raw[_HEADER.size:start])
        floats = _unshuffle(raw[start:start+8*nf], '<f8')
        integers = _unshuffle(raw[start+8*nf:], '<i8')
        return [_decode(tree, floats, integers) for tree in trees]
    except (zlib.error, UnicodeError, json.JSONDecodeError, IndexError, KeyError, TypeError, OverflowError, struct.error) as error:
        raise ValueError('Malformed native raw chunk') from error


def _validate_record(record, previous):
    if type(record) is not dict:
        raise ValueError('Raw record must be a dictionary')
    wrapped = 'kind' in record
    actual = record.get('actual') if wrapped else record
    if type(actual) is not dict or actual.get('schema') != 'dual-raw-v1':
        raise ValueError('Exact dual-raw-v1 schema required; stale/mixed raw is forbidden')
    world_time = actual.get('world_time')
    if type(world_time) not in (int, float) or not math.isfinite(world_time) or world_time < 0:
        raise ValueError('Finite nonnegative world_time required')
    if wrapped:
        if record['kind'] not in ('RESET', 'STEP'):
            raise ValueError('Native wrapper kind must be RESET or STEP')
        if previous is None and record['kind'] != 'RESET':
            raise ValueError('Native probe archive must begin with RESET')
        if previous is not None and record['kind'] == 'RESET':
            raise ValueError('Duplicate RESET is not a monotone single mission')
        tick = record.get('tick', 0 if record['kind'] == 'RESET' else None)
        tick_mode = True
    else:
        tick = record.get('tick')
        tick_mode = 'tick' in record
    if tick_mode and (type(tick) is not int or tick < 0):
        raise ValueError('Explicit nonnegative integer tick required')
    identities = tuple((k, record.get(k, actual.get(k))) for k in _IDENTITIES)
    if previous is not None:
        if wrapped != previous['wrapped'] or identities != previous['identities']:
            raise ValueError('Mixed archive record shape or mission/model identity')
        if tick_mode != previous['tick_mode']:
            raise ValueError('Tick presence must remain consistent')
        if world_time <= previous['world_time'] or (tick_mode and tick <= previous['tick']):
            raise ValueError('Record tick and world_time must be strictly monotone and unique')
    return dict(world_time=world_time, tick=tick, tick_mode=tick_mode, wrapped=wrapped, identities=identities)


class RawWriter:
    def __init__(self, path, *, chunk_frames=16, compression_level=3):
        if type(chunk_frames) is not int or chunk_frames < 1 or type(compression_level) is not int or not 0 <= compression_level <= 9:
            raise ValueError('Positive chunk size and compression level0..9 required')
        self.path = Path(path)
        self.stream = self.path.open('xb')  # Never overwrite original or prior evidence.
        self.stream.write(_MAGIC)
        self.chunk_frames, self.compression_level = chunk_frames, compression_level
        self.trees, self.floats, self.integers = [], array('d'), array('q')
        self.previous, self.chunks, self.frame_count = None, [], 0
        self.digest, self.closed = sha256(_MAGIC), False

    def append(self, record):
        if self.closed:
            raise ValueError('Raw writer is closed')
        state = _validate_record(record, self.previous)
        nf, ni = len(self.floats), len(self.integers)
        try:
            tree = _encode(record, self.floats, self.integers)
        except (ValueError, OverflowError, RecursionError) as error:
            del self.floats[nf:]
            del self.integers[ni:]
            raise ValueError('Invalid native JSON record') from error
        self.trees.append(tree)
        self.previous = state
        self.frame_count += 1
        if len(self.trees) >= self.chunk_frames:
            self._flush()

    def _flush(self):
        if not self.trees:
            return
        payload = _pack(self.trees, self.floats, self.integers, self.compression_level)
        digest = sha256(payload).digest()
        offset = self.stream.tell()
        block = b'C'+_U64.pack(len(payload))+digest+payload
        self.stream.write(block)
        self.digest.update(block)
        self.chunks.append(dict(offset=offset, count=len(self.trees), sha256=digest.hex()))
        self.trees, self.floats, self.integers = [], array('d'), array('q')

    def close(self):
        if self.closed:
            return
        try:
            self._flush()
            footer = dict(schema='dual-raw-storage-v1', record_schema='dual-raw-v1', frame_count=self.frame_count,
                          chunk_count=len(self.chunks), chunks=self.chunks, content_sha256=self.digest.hexdigest())
            payload = json.dumps(footer, separators=(',', ':'), allow_nan=False).encode('utf-8')
            self.stream.write(b'F'+_U64.pack(len(payload))+sha256(payload).digest()+payload)
            self.stream.flush()
            os.fsync(self.stream.fileno())
        finally:
            self.stream.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if kind is None:
            self.close()
        else:
            # An interrupted mission must not acquire a success-looking footer.
            self.stream.close()
            self.closed = True


def _read_exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise ValueError('Truncated native raw archive')
    return data


def _read_block(stream):
    offset = stream.tell()
    tag = stream.read(1)
    if tag not in (b'C', b'F'):
        raise ValueError('Mandatory native raw footer missing or unknown block')
    size_bytes = _read_exact(stream, 8)
    size = _U64.unpack(size_bytes)[0]
    if size > _MAX_BLOCK:
        raise ValueError('Raw block exceeds supported size')
    expected = _read_exact(stream, 32)
    payload = _read_exact(stream, size)
    if sha256(payload).digest() != expected:
        raise ValueError('Native raw checksum mismatch')
    return tag, offset, expected.hex(), payload, tag+size_bytes+expected


def _scan(stream):
    if _read_exact(stream, len(_MAGIC)) != _MAGIC:
        raise ValueError('Unsupported native raw storage schema')
    digest, chunks, count, previous = sha256(_MAGIC), [], 0, None
    while True:
        tag, offset, checksum, payload, header = _read_block(stream)
        if tag == b'C':
            rows = _unpack(payload)
            if not rows:
                raise ValueError('Empty data chunk')
            for row in rows:
                previous = _validate_record(row, previous)
            chunks.append(dict(offset=offset, count=len(rows), sha256=checksum))
            count += len(rows)
            digest.update(header)
            digest.update(payload)
            continue
        try:
            footer = json.loads(payload)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError('Malformed native raw footer') from error
        expected = dict(schema='dual-raw-storage-v1', record_schema='dual-raw-v1', frame_count=count,
                        chunk_count=len(chunks), chunks=chunks, content_sha256=digest.hexdigest())
        if footer != expected or stream.read(1):
            raise ValueError('Native raw footer index/count/hash mismatch or trailing bytes')
        return dict(footer, footer_offset=offset)


def inspect_raw(path):
    """Fully validate archive, returning counts/index/checksum footer metadata."""
    with Path(path).open('rb') as stream:
        return _scan(stream)


def read_raw(path):
    """Yield exact dictionaries, only after whole-file integrity/semantic checks."""
    with Path(path).open('rb') as stream:
        footer = _scan(stream)
        for chunk in footer['chunks']:
            stream.seek(chunk['offset'])
            tag, _, checksum, payload, _ = _read_block(stream)
            if tag != b'C' or checksum != chunk['sha256']:
                raise ValueError('Native raw changed during verified read')
            yield from _unpack(payload)
