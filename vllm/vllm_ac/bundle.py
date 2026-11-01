"""Compact container for independently decodable arithmetic chunks.

Deflate shares repeated metadata across chunk archives. It does not replace the
arithmetic payload or store source text, target tokens, or probability tables.
"""
import hashlib
import struct
import zlib
from pathlib import Path

MAGIC = b'VAB1'


def pack_bundle(archives):
    """Return one checksummed container from ordered VAC archives or results."""
    records = [item.archive if hasattr(item, 'archive') else bytes(item) for item in archives]
    body = struct.pack('>I', len(records)) + b''.join(
        struct.pack('>I', len(record)) + record for record in records)
    return MAGIC + hashlib.sha256(body).digest() + zlib.compress(body)


def unpack_bundle(bundle):
    if len(bundle) < 36 or bundle[:4] != MAGIC:
        raise ValueError('Invalid arithmetic bundle')
    try:
        decoder = zlib.decompressobj()
        body = decoder.decompress(bundle[36:]) + decoder.flush()
    except zlib.error as exc:
        raise ValueError('Corrupt arithmetic bundle') from exc
    if not decoder.eof or decoder.unused_data or hashlib.sha256(body).digest() != bundle[4:36]:
        raise ValueError('Corrupt or truncated arithmetic bundle')
    if len(body) < 4:
        raise ValueError('Missing chunk count')
    count = struct.unpack_from('>I', body)[0]
    offset, records = 4, []
    for _ in range(count):
        if offset + 4 > len(body):
            raise ValueError('Truncated chunk length')
        size = struct.unpack_from('>I', body, offset)[0]
        offset += 4
        if size > len(body) - offset:
            raise ValueError('Truncated chunk archive')
        records.append(body[offset:offset+size])
        offset += size
    if offset != len(body):
        raise ValueError('Unexpected trailing bundle bytes')
    return records


def save_bundle(archives, path):
    with Path(path).open('xb') as stream:
        stream.write(pack_bundle(archives))


def decompress_bundle(bundle, decode):
    """decode is e.g. lambda archive: decompress(llm, archive)."""
    return ''.join(decode(archive) for archive in unpack_bundle(bundle))


COMPACT_CHUNK_TOKENS = 1024


def pack_compact(payloads, final_token_count):
    """Externally configured format: u16 final count, repeated u16 size/payload."""
    payloads = [bytes(payload) for payload in payloads]
    if not isinstance(final_token_count, int) or isinstance(final_token_count, bool):
        raise ValueError('Final token count must be an integer')
    if (payloads and not 1 <= final_token_count <= COMPACT_CHUNK_TOKENS
            or not payloads and final_token_count != 0):
        raise ValueError('Inconsistent compact final token count')
    if any(not 1 <= len(payload) <= 65535 for payload in payloads):
        raise ValueError('Compact payload length must be between 1 and 65535 bytes')
    return struct.pack('>H', final_token_count) + b''.join(
        struct.pack('>H', len(payload)) + payload for payload in payloads)


def unpack_compact(archive):
    """Parse through EOF; no magic, checksum, or embedded execution profile."""
    if len(archive) < 2:
        raise ValueError('Missing compact final token count')
    final_count = struct.unpack_from('>H', archive)[0]
    offset, payloads = 2, []
    while offset < len(archive):
        if offset + 2 > len(archive):
            raise ValueError('Truncated compact payload length')
        size = struct.unpack_from('>H', archive, offset)[0]
        offset += 2
        if size == 0 or size > len(archive) - offset:
            raise ValueError('Invalid or truncated compact payload')
        payloads.append(archive[offset:offset + size])
        offset += size
    if (payloads and not 1 <= final_count <= COMPACT_CHUNK_TOKENS
            or not payloads and final_count != 0):
        raise ValueError('Inconsistent compact final token count')
    return final_count, payloads
