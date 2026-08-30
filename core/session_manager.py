import asyncio
import json
import os
import time
from typing import Dict, Optional, List, Any
from loguru import logger

from core.session import ClientSession
from config.settings import AppConfig


class SessionManager:
    """
    Manages active ClientSessions, enforces connection limits,
    tracks online metrics, and handles session lifecycle including
    disconnect grace-period resume and transcript persistence.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._sessions: Dict[str, ClientSession] = {}
        self._lock = asyncio.Lock()
        # Strong references to background finalization tasks (loop-only refs
        # are not enough to guarantee completion)
        self._bg_finalize_tasks: set = set()

    @property
    def active_session_count(self) -> int:
        return len(self._sessions)

    async def create_session(
        self,
        session_id: Optional[str] = None,
        language: str = "zh",
        enable_spk: bool = True,
        expected_speakers: int = 2,
        hotwords: Optional[List[str]] = None,
    ) -> ClientSession:
        async with self._lock:
            if len(self._sessions) >= self.config.server.max_connections:
                raise RuntimeError(
                    f"Max concurrent connections limit ({self.config.server.max_connections}) reached."
                )

            session = ClientSession(
                session_id=session_id,
                language=language,
                enable_spk=enable_spk,
                expected_speakers=expected_speakers,
                hotwords=hotwords,
                app_config=self.config,
            )

            if session.session_id in self._sessions:
                # Clean up existing old session with same ID if any
                old_session = self._sessions[session.session_id]
                old_session.cleanup()

            self._sessions[session.session_id] = session
            logger.info(
                f"[SessionManager] Created session {session.session_id}, total active: {len(self._sessions)}"
            )
            return session

    async def get_session(self, session_id: str) -> Optional[ClientSession]:
        async with self._lock:
            return self._sessions.get(session_id)

    async def detach_session(self, session_id: str, pipeline: Any):
        """
        Mark a session as detached after an unexpected disconnect, keeping it
        resumable for the configured grace period. The pipeline's output queue
        is swapped to a fresh queue with limited size so finalization of
        in-flight jobs can never block on the dead connection's bounded queue,
        but also does not grow unbounded.
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.is_stopping:
                return False
            session.detached_at = time.time()
            session.detached_pipeline = pipeline
            session.is_active = False
            # Use bounded queue to prevent unbounded growth during detached period;
            # final results are still persisted via session.sentences
            pipeline.output_queue = asyncio.Queue(maxsize=1024)
            logger.info(
                f"[SessionManager] Session {session_id} detached, resumable for "
                f"{self.config.server.resume_ttl_sec}s."
            )
            return True

    async def reattach_session(self, session_id: str) -> Optional[ClientSession]:
        """
        Reactivate a detached session for a resuming client, if the resume
        grace period has not expired. An expired-but-not-yet-cleaned session
        is finalized in the background and None is returned, so the caller
        falls through to creating a brand-new session.
        """
        expired: Optional[ClientSession] = None
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.detached_at is None:
                return None
            if time.time() - session.detached_at > self.config.server.resume_ttl_sec:
                expired = session
                self._sessions.pop(session_id, None)
            else:
                session.detached_at = None
                session.detached_pipeline = None
                session.is_active = True
                session.touch()
                logger.info(f"[SessionManager] Session {session_id} reattached (resumed).")
                return session

        if expired is not None:
            # Finalize without blocking the new connection setup
            self._spawn_finalize(expired)
        return None

    async def remove_session(self, session_id: str):
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session:
            # File I/O off the event loop (long transcripts on slow disks)
            await asyncio.get_running_loop().run_in_executor(
                None, self._persist_transcript, session
            )
            session.cleanup()
            logger.info(
                f"[SessionManager] Removed session {session_id}, total active: {len(self._sessions)}"
            )

    async def list_sessions(self) -> List[Dict[str, Any]]:
        async with self._lock:
            result = []
            now = time.time()
            for s_id, sess in self._sessions.items():
                result.append(
                    {
                        "session_id": s_id,
                        "created_at": sess.created_at,
                        "last_active_at": sess.last_active_at,
                        "idle_seconds": round(now - sess.last_active_at, 2),
                        "sentence_count": sess.sentence_seq,
                        "is_in_speech": sess.is_in_speech,
                        "detached": sess.detached_at is not None,
                    }
                )
            return result

    async def cleanup_idle_sessions(self, max_idle_sec: Optional[float] = None):
        """
        Periodic maintenance:
        - Detached sessions whose resume grace period expired are finalized
          (flush + re-cluster via their detached pipeline) and removed.
        - Sessions idle beyond max_idle_sec are evicted.
        """
        if max_idle_sec is None:
            max_idle_sec = getattr(self.config.server, "idle_timeout_sec", 300.0)
        resume_ttl = self.config.server.resume_ttl_sec
        now = time.time()

        expired_detached: List[ClientSession] = []
        idle_sessions: List[ClientSession] = []
        async with self._lock:
            for s_id, sess in self._sessions.items():
                if sess.detached_at is not None:
                    if now - sess.detached_at > resume_ttl:
                        expired_detached.append(sess)
                elif now - sess.last_active_at > max_idle_sec:
                    idle_sessions.append(sess)

            for sess in expired_detached + idle_sessions:
                self._sessions.pop(sess.session_id, None)

        # Finalize outside the lock: process_stop may wait for in-flight jobs
        for sess in expired_detached:
            self._spawn_finalize(sess)

        for sess in idle_sessions:
            logger.warning(f"[SessionManager] Evicting idle session {sess.session_id} (idle {now - sess.last_active_at:.1f}s > {max_idle_sec}s)")
            try:
                from core.metrics.prometheus_metrics import metrics

                metrics.inc_counter("asr_sessions_evicted_idle")
            except Exception:
                pass
            await asyncio.get_running_loop().run_in_executor(
                None, self._persist_transcript, sess
            )
            sess.cleanup()

    def _spawn_finalize(self, session: ClientSession):
        """Finalize an expired detached session as a tracked background task."""
        task = asyncio.create_task(self._finalize_detached(session))
        self._bg_finalize_tasks.add(task)
        task.add_done_callback(self._bg_finalize_tasks.discard)

    async def _finalize_detached(self, session: ClientSession):
        logger.warning(
            f"[SessionManager] Resume window expired for session {session.session_id}, finalizing."
        )
        pipeline = session.detached_pipeline
        if pipeline is not None:
            try:
                await asyncio.wait_for(
                    pipeline.process_stop(),
                    timeout=self.config.final_asr.hard_timeout_sec + 2.0,
                )
            except Exception as e:
                logger.error(
                    f"[SessionManager] Finalizing detached session {session.session_id} failed: {e}"
                )
        await asyncio.get_running_loop().run_in_executor(
            None, self._persist_transcript, session
        )
        session.cleanup()

    def _persist_transcript(self, session: ClientSession):
        transcript_dir = self.config.observability.transcript_dir
        if not transcript_dir:
            return
        try:
            os.makedirs(transcript_dir, exist_ok=True)
            payload = {
                "session_id": session.session_id,
                "created_at": session.created_at,
                "closed_at": time.time(),
                "duration_ms": session.ring_buffer.total_ms,
                "total_sentences": len(session.sentences),
                "sentences": session.commit_transcript(),
            }
            path = os.path.join(transcript_dir, f"{session.session_id}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            logger.info(f"[SessionManager] Transcript persisted to {path}")
        except Exception as e:
            logger.error(f"[SessionManager] Failed to persist transcript: {e}")
