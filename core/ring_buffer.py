import threading
from typing import Optional

from loguru import logger


class AudioRingBuffer:
    """
    Circular audio ring buffer for raw PCM16 audio (16kHz, mono, 2 bytes/sample).
    Maintains continuous absolute audio timeline (ms) and supports pre-roll extraction.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        bytes_per_sample: int = 2,
        max_duration_sec: float = 30.0,
        pre_roll_ms: int = 800,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.bytes_per_sample = bytes_per_sample
        self.bytes_per_ms = (sample_rate * channels * bytes_per_sample) // 1000
        self.max_bytes = int(max_duration_sec * 1000 * self.bytes_per_ms)
        self.pre_roll_ms = pre_roll_ms
        self.pre_roll_bytes = pre_roll_ms * self.bytes_per_ms

        self._buffer = bytearray(self.max_bytes)
        self._head = 0  # Write index in circular buffer
        self._total_bytes_written = 0
        self._lock = threading.Lock()

    @property
    def total_ms(self) -> int:
        with self._lock:
            return self._total_bytes_written // self.bytes_per_ms

    @property
    def total_bytes_written(self) -> int:
        with self._lock:
            return self._total_bytes_written

    def write(self, data: bytes) -> int:
        """
        Write new PCM audio chunks into the ring buffer.
        Returns the new total written milliseconds.
        """
        if not data:
            return self.total_ms

        with self._lock:
            n = len(data)
            # If incoming chunk is larger than buffer, keep only the tail
            if n >= self.max_bytes:
                data = data[-self.max_bytes:]
                n = len(data)

            space_to_end = self.max_bytes - self._head
            if n <= space_to_end:
                self._buffer[self._head : self._head + n] = data
                self._head = (self._head + n) % self.max_bytes
            else:
                self._buffer[self._head : self.max_bytes] = data[:space_to_end]
                rem = n - space_to_end
                self._buffer[0:rem] = data[space_to_end:]
                self._head = rem

            self._total_bytes_written += n
            return self._total_bytes_written // self.bytes_per_ms

    def get_range(self, start_ms: int, end_ms: int) -> bytes:
        """
        Extract audio between start_ms and end_ms (absolute timestamps).
        """
        with self._lock:
            total_bytes = self._total_bytes_written
            current_total_ms = total_bytes // self.bytes_per_ms
            oldest_ms = max(0, (total_bytes - self.max_bytes) // self.bytes_per_ms)

            # Clamp boundaries
            clamped_start_ms = max(oldest_ms, min(start_ms, current_total_ms))
            clamped_end_ms = max(clamped_start_ms, min(end_ms, current_total_ms))

            if clamped_start_ms != start_ms or clamped_end_ms != end_ms:
                logger.warning(
                    f"[AudioRingBuffer] Clamped range [{start_ms},{end_ms}) -> "
                    f"[{clamped_start_ms},{clamped_end_ms}) (oldest={oldest_ms}, total={current_total_ms})"
                )

            if clamped_start_ms >= clamped_end_ms:
                return b""

            start_byte_offset = clamped_start_ms * self.bytes_per_ms
            end_byte_offset = clamped_end_ms * self.bytes_per_ms
            length_to_read = end_byte_offset - start_byte_offset

            # Calculate index in circular buffer
            # The current head corresponds to total_bytes
            # Distance from current total_bytes backwards
            dist_from_tail = total_bytes - start_byte_offset
            read_start_idx = (self._head - dist_from_tail) % self.max_bytes

            if read_start_idx + length_to_read <= self.max_bytes:
                return bytes(self._buffer[read_start_idx : read_start_idx + length_to_read])
            else:
                part1_len = self.max_bytes - read_start_idx
                part2_len = length_to_read - part1_len
                return bytes(self._buffer[read_start_idx : self.max_bytes] + self._buffer[0:part2_len])

    def get_segment_with_preroll(
        self, start_ms: int, end_ms: int, custom_preroll_ms: Optional[int] = None
    ) -> bytes:
        """
        Extract audio segment for VAD speech interval [start_ms, end_ms]
        with pre-roll included to prevent clipped word onsets.
        """
        preroll = self.pre_roll_ms if custom_preroll_ms is None else custom_preroll_ms
        effective_start_ms = max(0, start_ms - preroll)
        return self.get_range(effective_start_ms, end_ms)

    def clear(self):
        with self._lock:
            self._head = 0
            self._total_bytes_written = 0
            self._buffer = bytearray(self.max_bytes)
