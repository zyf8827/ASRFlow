import unittest

from aiohttp import ClientSession

from config.settings import AppConfig
from core.session_manager import SessionManager
from server.http_server import HTTPServer


class TestHTTPDashboard(unittest.IsolatedAsyncioTestCase):
    """The console page (functional test + metrics) and existing HTTP endpoints."""

    async def asyncSetUp(self):
        self.config = AppConfig()
        self.config.server.host = "127.0.0.1"
        self.config.server.http_port = 18196  # distinct from e2e ports 10199/10198
        self.session_manager = SessionManager(self.config)
        self.http_server = HTTPServer(self.session_manager, self.config)
        await self.http_server.start()
        self.base_url = f"http://127.0.0.1:{self.config.server.http_port}"

    async def asyncTearDown(self):
        await self.http_server.stop()

    async def _get(self, path: str):
        async with ClientSession() as client:
            async with client.get(self.base_url + path) as resp:
                return resp.status, resp.content_type, await resp.text()

    async def test_dashboard_served_at_root(self):
        status, ctype, body = await self._get("/")
        self.assertEqual(status, 200)
        self.assertEqual(ctype, "text/html")
        self.assertIn("ASRFlow", body)
        # Functional-test tab: real-time ASR via WebSocket + mic/file capture
        self.assertIn("WebSocket", body)
        self.assertIn("getUserMedia", body)
        self.assertIn("decodeAudioData", body)
        # WS frame monitor for diagnostics (all up/down frames, filterable)
        self.assertIn("WS 帧监控", body)
        self.assertIn("ws-log", body)
        # Metrics tab: vanilla JS polling the existing endpoints
        for endpoint in ("/healthz", "/ready", "/metrics", "/api/v1/sessions"):
            self.assertIn(endpoint, body)
        # Self-contained page: no CDN / third-party JS dependencies
        self.assertNotIn('src="http', body)
        self.assertNotIn('href="http', body)

    async def test_dashboard_served_at_dashboard_alias(self):
        status, ctype, _ = await self._get("/dashboard")
        self.assertEqual(status, 200)
        self.assertEqual(ctype, "text/html")

    async def test_ready_exposes_ws_port_for_dashboard_default_url(self):
        status, _, body = await self._get("/ready")
        self.assertEqual(status, 200)
        self.assertIn('"ws_port"', body)

    async def test_existing_endpoints_still_work(self):
        status, _, body = await self._get("/healthz")
        self.assertEqual(status, 200)
        self.assertIn("uptime_seconds", body)

        status, _, body = await self._get("/metrics")
        self.assertEqual(status, 200)
        self.assertIn("asr_online_sessions", body)


class TestReadinessProbes(unittest.IsolatedAsyncioTestCase):
    """/ready must reflect engine health and admission state."""

    async def _start_server(self, readiness_probes=None, admission=None, port=18197):
        config = AppConfig()
        config.server.host = "127.0.0.1"
        config.server.http_port = port
        server = HTTPServer(
            SessionManager(config),
            config,
            readiness_probes=readiness_probes,
            admission=admission,
        )
        await server.start()
        return server, f"http://127.0.0.1:{config.server.http_port}"

    async def test_ready_503_when_engine_probe_fails(self):
        server, base = await self._start_server(
            readiness_probes={"streaming": lambda: False, "vad": lambda: True}
        )
        try:
            async with ClientSession() as client:
                async with client.get(base + "/ready") as resp:
                    self.assertEqual(resp.status, 503)
                    body = await resp.json()
                    self.assertFalse(body["ready"])
                    self.assertFalse(body["engines"]["streaming"])
                    self.assertTrue(body["engines"]["vad"])
        finally:
            await server.stop()

    async def test_ready_200_with_healthy_probes_and_engine_status(self):
        server, base = await self._start_server(
            readiness_probes={"streaming": lambda: True, "vad": lambda: True}
        )
        try:
            async with ClientSession() as client:
                async with client.get(base + "/ready") as resp:
                    self.assertEqual(resp.status, 200)
                    body = await resp.json()
                    self.assertTrue(body["engines"]["streaming"])
        finally:
            await server.stop()

    async def test_ready_reports_admission_info_without_flipping_ready(self):
        class FakeAdmission:
            saturated = True
            reasons = ["streaming_wait_ms=999.0>200"]

        server, base = await self._start_server(admission=FakeAdmission())
        try:
            async with ClientSession() as client:
                async with client.get(base + "/ready") as resp:
                    self.assertEqual(resp.status, 200)
                    body = await resp.json()
                    self.assertTrue(body["ready"])
                    self.assertTrue(body["admission"]["saturated"])
                    self.assertIn("streaming_wait_ms", body["admission"]["reasons"][0])
        finally:
            await server.stop()


if __name__ == "__main__":
    unittest.main()
