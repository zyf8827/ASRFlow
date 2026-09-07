"""强制切句边界上的零长 COMMIT/STOP 回归测试。

背景 (线上实测): VAD 判定 speech_start=0ms 时, 30s 强制切句重臂 speech_start 后
紧跟客户端 COMMIT, process_commit 会切出 [t~t] 零长段; 该段此前会 (a) 触发
ONNX 引擎对空 lfr 缓存的 final flush -> funasr 前端 np.stack([]) 崩溃, 连接被
带断 ("need at least one array to stack"); (b) 空段进入二遍浪费调用。
"""
import asyncio
import unittest

from config.settings import AppConfig
from core.ring_buffer import AudioRingBuffer
from core.session import ClientSession
from pipeline.session_pipeline import SessionPipeline


class _StubVAD:
    def process_chunk(self, audio_bytes, cache, is_final=False):
        return []


def _make_pipeline(session):
    """只装配 process_commit/process_stop 零长路径触达的最小依赖。

    thread_pool=None: asyncio 默认执行器即可跑 stub VAD。
    """
    pipe = object.__new__(SessionPipeline)
    pipe.session = session
    pipe.vad_engine = _StubVAD()
    pipe.config = AppConfig()
    pipe.output_queue = asyncio.Queue()
    pipe.thread_pool = None
    return pipe


def _make_session_at_force_cut_boundary():
    """模拟强制切句后的会话态: speech 已重臂到当前时间戳, 缓存已清, 无新音频。"""
    cfg = AppConfig()
    session = ClientSession(session_id="t-zero", app_config=cfg)
    session.is_active = True
    session.ring_buffer.write(b"\x00\x00" * 16000 * 2)  # 2s 音频
    session.is_in_speech = True                         # 强制切句保持 speech 开启
    session.speech_start_ms = session.ring_buffer.total_ms  # 重臂到当前时间戳
    session.current_partial_text = ""                   # 切句 flush 已清空
    session.vad_cache.clear()
    return session


class TestZeroLengthCommit(unittest.TestCase):
    def test_commit_at_force_cut_boundary_is_noop(self):
        async def run():
            session = _make_session_at_force_cut_boundary()
            pipe = _make_pipeline(session)
            boundary_ms = session.ring_buffer.total_ms
            await pipe.process_commit()
            return session, boundary_ms

        session, boundary_ms = asyncio.run(run())
        # 不产生任何句记录 (无二遍调用), 状态正常复位, 水位推进
        self.assertEqual(session.sentences, {})
        self.assertFalse(session.is_in_speech)
        self.assertEqual(session.speech_start_ms, -1)
        self.assertGreaterEqual(session.last_committed_end_ms, boundary_ms)

    def test_stop_right_after_force_cut_boundary_skips_empty_final(self):
        async def run():
            session = _make_session_at_force_cut_boundary()
            pipe = _make_pipeline(session)
            await pipe.process_stop()
            # session_finished 正常发出 (收尾链路未受零长段影响)
            msg = await asyncio.wait_for(pipe.output_queue.get(), timeout=2)
            return session, msg

        session, msg = asyncio.run(run())
        self.assertEqual(session.sentences, {})
        self.assertFalse(session.is_in_speech)
        self.assertTrue(session.is_stopping)
        self.assertEqual(msg.get("type"), "session_finished")

    def test_commit_with_audio_still_works(self):
        """非零长 COMMIT (有已积累 partial 文本) 不受守卫影响, 仍走正常收句。

        正常路径会进入 _handle_sentence_endpoint 的引擎调用 (stub 缺真实引擎,
        以 AttributeError 告终); 若被零长守卫误伤则会静默返回且无任何异常。
        """

        async def run():
            cfg = AppConfig()
            session = ClientSession(session_id="t-normal", app_config=cfg)
            session.is_active = True
            session.ring_buffer.write(b"\x00\x00" * 16000 * 2)
            session.is_in_speech = True
            session.speech_start_ms = 0
            session.current_partial_text = "测试文本"
            pipe = _make_pipeline(session)
            with self.assertRaises(AttributeError):
                await pipe.process_commit()
            return session

        session = asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
