"""Framed arithmetic streams. No target IDs or probability tables in archives."""
import hashlib
import io
import json
import struct
from dataclasses import dataclass
from pathlib import Path

from .arithmeticcoding import ArithmeticEncoder, ArithmeticDecoder, BitInputStream, BitOutputStream
from .cdf import TOTAL

MAGIC = b'VAC2'


class _Int(int):
    def item(self):
        return int(self)


def interval_table(low, high):
    if not 0 <= low < high <= TOTAL:
        raise ValueError('Invalid arithmetic interval')
    return [_Int(low), _Int(high), _Int(TOTAL)]


class Encoder:
    def __init__(self):
        self.stream = io.BytesIO()
        self.bits = BitOutputStream(self.stream)
        self.coder = ArithmeticEncoder(32, self.bits)

    def write(self, low, high):
        self.coder.write(interval_table(low, high), 0)

    def finish(self):
        self.coder.finish()
        while self.bits.numbitsfilled:
            self.bits.write(0)
        return self.stream.getvalue()


class Decoder:
    def __init__(self, payload):
        self.coder = ArithmeticDecoder(32, BitInputStream(io.BytesIO(payload)))

    def value(self):
        c = self.coder
        return ((c.code - c.low + 1) * TOTAL - 1) // (c.high - c.low + 1)

    def advance(self, low, high):
        if not low <= self.value() < high:
            raise ValueError('Decoded interval does not contain code value')
        self.coder.update(interval_table(low, high), 0)


@dataclass
class CompressionResult:
    archive: bytes
    metadata: dict
    payload_bytes: int

    def save(self, path):
        with Path(path).open('xb') as stream:
            stream.write(self.archive)


def pack(payload, metadata):
    metadata = dict(metadata, format='vllm-ac-v2', cdf='softmax-fp32-weights30-cdf24-v2',
                    payload_sha256=hashlib.sha256(payload).hexdigest())
    header = json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    archive = MAGIC + struct.pack('>I', len(header)) + header + payload
    return CompressionResult(archive, metadata, len(payload))


def unpack(archive):
    if len(archive) < 8 or archive[:4] != MAGIC:
        raise ValueError('Invalid arithmetic archive')
    size = struct.unpack('>I', archive[4:8])[0]
    if size > len(archive) - 8:
        raise ValueError('Truncated archive header')
    meta = json.loads(archive[8:8 + size])
    payload = archive[8 + size:]
    if meta.get('format') != 'vllm-ac-v2' or meta.get('cdf') != 'softmax-fp32-weights30-cdf24-v2':
        raise ValueError('Unsupported archive format')
    if hashlib.sha256(payload).hexdigest() != meta['payload_sha256']:
        raise ValueError('Corrupt or truncated arithmetic payload')
    if not isinstance(meta['token_count'], int) or meta['token_count'] < 0:
        raise ValueError('Invalid token count')
    return meta, payload


def text_metadata(text, ids, prompt_ids, **extra):
    raw = text.encode('utf-8')
    return dict(extra, prompt_token_ids=prompt_ids, token_count=len(ids),
                source_bytes=len(raw), source_sha256=hashlib.sha256(raw).hexdigest())


def recover_text(tokenizer, ids, meta):
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    raw = text.encode('utf-8')
    if len(raw) != meta['source_bytes'] or hashlib.sha256(raw).hexdigest() != meta['source_sha256']:
        raise ValueError('Reconstruction checksum mismatch: model/CDF execution is not reproducible')
    return text
