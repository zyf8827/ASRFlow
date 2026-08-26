"""Session pipeline stub — wires VAD / streaming ASR / final ASR later."""
from __future__ import annotations

from typing import Any, Callable, Optional


class SessionPipeline:
    """Minimal pipeline placeholder for early integration."""

    def __init__(self, session_id: str, on_event: Optional[Callable[[dict], Any]] = None):
        self.session_id = session_id
        self.on_event = on_event
        self._closed = False

    async def feed_audio(self, pcm: bytes) -> None:
        if self._closed:
            return
        # Stub: no events yet
        return None

    async def close(self) -> None:
        self._closed = True
