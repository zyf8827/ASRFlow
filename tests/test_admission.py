import json
import asyncio
import unittest
import websockets

from config.settings import AdmissionConfig
from server.admission import AdmissionController
from tests.test_websocket_e2e import _MockServiceHarness
from tests.test_websocket_e2e import generate_synthetic_speech_pcm


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, sec: float):
        self.now += sec


def _make_controller(value: float, clock, **overrides):
    cfg = AdmissionConfig(**{"saturate_sec": 5.0, "recover_sec": 10.0, **overrides})
    holder = {"v": value}
    probes = {"streaming_wait_ms": (lambda: holder["v"], cfg.streaming_wait_ms_limit)}
    return AdmissionController(cfg, probes, clock=clock), holder


class TestAdmissionStateMachine(unittest.TestCase):
    def test_below_limit_never_saturates(self):
        clock = _FakeClock()
        ctl, _ = _make_controller(50.0, clock)  # limit 200
        for _ in range(30):
            ctl.evaluate()
            clock.advance(1.0)
        self.assertTrue(ctl.allow_new_session()[0])

    def test_saturates_only_after_saturate_sec(self):
        clock = _FakeClock()
        ctl, _ = _make_controller(999.0, clock)
        ctl.evaluate()  # over_since = 0
        clock.advance(4.9)
        ctl.evaluate()
        self.assertTrue(ctl.allow_new_session()[0], "must not saturate before saturate_sec")
        clock.advance(0.2)
        saturated, reasons = ctl.evaluate()
        self.assertTrue(saturated)
        ok, reasons = ctl.allow_new_session()
        self.assertFalse(ok)
        self.assertTrue(any("streaming_wait_ms" in r for r in reasons))

    def test_flapping_below_saturate_sec_never_saturates(self):
        clock = _FakeClock()
        ctl, holder = _make_controller(999.0, clock)
        for i in range(10):
            holder["v"] = 999.0 if i % 2 == 0 else 10.0
            ctl.evaluate()
            clock.advance(3.0)  # over-streak never reaches 5s
        self.assertTrue(ctl.allow_new_session()[0])

    def test_recovery_requires_below_half_for_recover_sec(self):
        clock = _FakeClock()
        ctl, holder = _make_controller(999.0, clock)
        ctl.evaluate()
        clock.advance(6.0)
        ctl.evaluate()
        self.assertTrue(ctl.saturated)

        holder["v"] = 120.0  # below limit(200) but above half(100): no recovery
        clock.advance(60.0)
        ctl.evaluate()
        self.assertTrue(ctl.saturated, "must stay saturated while above half threshold")

        holder["v"] = 40.0  # below half
        ctl.evaluate()  # under_since = now
        clock.advance(9.0)
        ctl.evaluate()
        self.assertTrue(ctl.saturated, "must stay saturated before recover_sec")
        clock.advance(1.5)
        ctl.evaluate()
        self.assertFalse(ctl.saturated)
        self.assertTrue(ctl.allow_new_session()[0])

    def test_disabled_always_allows(self):
        clock = _FakeClock()
        ctl, _ = _make_controller(999.0, clock, enable=False)
        ctl.evaluate()
        clock.advance(60.0)
        ctl.evaluate()
        self.assertTrue(ctl.allow_new_session()[0])


class _SaturatedAdmissionHarness(_MockServiceHarness):
    """Gateway wired with an already-saturated admission controller."""

    async def asyncSetUp(self):
        # set before super() so the harness constructs the gateway with it
        self.admission = AdmissionController(
            AdmissionConfig(saturate_sec=0.0, recover_sec=0.0),
            {"streaming_wait_ms": (lambda: 999.0, 200.0)},
        )
        self.admission.evaluate()  # first tick records over_since...
        self.admission.evaluate()  # ...second tick saturates (saturate_sec=0)
        assert self.admission.saturated
        await super().asyncSetUp()


class TestAdmissionRejection(_SaturatedAdmissionHarness):
    async def test_json_start_rejected_with_4290_and_close_1013(self):
        uri = f"ws://127.0.0.1:{self.config.server.port}"
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps({"type": "start", "session_id": "rejected-1"}))
            err = json.loads(await asyncio.wait_for(ws.recv(), timeout=3.0))
            self.assertEqual(err.get("type"), "error")
            self.assertEqual(err.get("code"), 4290)
            with self.assertRaises(websockets.ConnectionClosed) as ctx:
                await asyncio.wait_for(ws.recv(), timeout=3.0)
            self.assertEqual(ctx.exception.code, 1013)

    async def test_plain_start_rejected(self):
        uri = f"ws://127.0.0.1:{self.config.server.port}"
        async with websockets.connect(uri) as ws:
            await ws.send("START")
            err = json.loads(await asyncio.wait_for(ws.recv(), timeout=3.0))
            self.assertEqual(err.get("code"), 4290)
            with self.assertRaises(websockets.ConnectionClosed):
                await asyncio.wait_for(ws.recv(), timeout=3.0)

    async def test_binary_auto_create_rejected(self):
        uri = f"ws://127.0.0.1:{self.config.server.port}"
        async with websockets.connect(uri) as ws:
            await ws.send(b"\x00" * 1920)
            err = json.loads(await asyncio.wait_for(ws.recv(), timeout=3.0))
            self.assertEqual(err.get("code"), 4290)
            with self.assertRaises(websockets.ConnectionClosed):
                await asyncio.wait_for(ws.recv(), timeout=3.0)

    async def test_existing_session_continues_streaming(self):
        """Saturation only gates NEW sessions; established ones keep working."""
        self.admission._saturated = False  # create a session first
        uri = f"ws://127.0.0.1:{self.config.server.port}"
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps({"type": "start", "session_id": "established-1"}))
            ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=3.0))
            self.assertEqual(ready.get("type"), "session_ready")

            self.admission._saturated = True  # instance becomes overloaded

            async def collector():
                async for msg in ws:
                    d = json.loads(msg)
                    if d.get("type") == "session_finished":
                        return d

            collect_task = asyncio.create_task(collector())
            audio = generate_synthetic_speech_pcm(duration_sec=1.0)
            silence = b"\x00" * 32000
            for data in (audio, silence):
                for offset in range(0, len(data), 1920):
                    await ws.send(data[offset : offset + 1920])
                    await asyncio.sleep(0.01)
            await ws.send(json.dumps({"type": "stop"}))
            finished = await asyncio.wait_for(collect_task, timeout=5.0)
            self.assertIsNotNone(finished)


class TestAdmissionResumeBypass(_SaturatedAdmissionHarness):
    async def test_resume_of_detached_session_bypasses_admission(self):
        """A resuming session already owns its capacity; saturation must not block it."""
        self.admission._saturated = False
        uri = f"ws://127.0.0.1:{self.config.server.port}"
        session_id = "resume-bypass-1"
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps({"type": "start", "session_id": session_id}))
            ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=3.0))
            self.assertEqual(ready.get("type"), "session_ready")
            # produce one sentence, then drop without STOP
            audio = generate_synthetic_speech_pcm(duration_sec=3.0)
            silence = b"\x00" * 32000
            for data in (audio, silence):
                for offset in range(0, len(data), 1920):
                    await ws.send(data[offset : offset + 1920])
                    await asyncio.sleep(0.01)
            await asyncio.sleep(0.5)
        await asyncio.sleep(0.5)  # let the server-side detach complete

        self.admission._saturated = True  # instance is now overloaded
        async with websockets.connect(uri) as ws2:
            await ws2.send(json.dumps({"type": "start", "session_id": session_id, "resume": True}))
            resumed = json.loads(await asyncio.wait_for(ws2.recv(), timeout=3.0))
            self.assertEqual(resumed.get("type"), "session_resumed")


if __name__ == "__main__":
    unittest.main()
