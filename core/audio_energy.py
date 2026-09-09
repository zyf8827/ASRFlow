"""PCM16 energy helper for the streaming hallucination guard (L0).

Pure Python on purpose: core.session imports this module unconditionally,
and mock-only deployments (requirements-base.txt) do not ship numpy.
"""

from __future__ import annotations

import math
import struct

DIGITAL_SILENCE_DBFS = -240.0
INT16_FULL_SCALE = 32768.0
BYTES_PER_SAMPLE = 2


def pcm16_rms_dbfs(audio_bytes: bytes) -> float:
    """Return RMS of PCM16 bytes in dBFS. All-zero / empty -> -240 dBFS."""
    if not audio_bytes or len(audio_bytes) < BYTES_PER_SAMPLE:
        return DIGITAL_SILENCE_DBFS
    n = len(audio_bytes) // BYTES_PER_SAMPLE
    samples = struct.unpack_from(f"<{n}h", audio_bytes)
    acc = 0
    for s in samples:
        acc += s * s
    mean = acc / float(n)
    if mean <= 0.0:
        return DIGITAL_SILENCE_DBFS
    rms = math.sqrt(mean)
    return 20.0 * math.log10(rms / INT16_FULL_SCALE)
