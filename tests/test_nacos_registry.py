import os
import json
import asyncio
import unittest
from aiohttp import web

from config.settings import AppConfig, RegistryConfig
from core.registry import NacosRegistry


class FakeNacosServer:
    """Minimal Nacos v1 OpenAPI stub recording every request it receives."""

    def __init__(self):
        self.calls = []          # (method, path, params_or_form)
        self.fail_next_register = 0
        self.app = web.Application()
        self.app.router.add_post("/nacos/v1/auth/login", self._login)
        self.app.router.add_post("/nacos/v1/ns/instance", self._register)
        self.app.router.add_put("/nacos/v1/ns/instance/beat", self._beat)
        self.app.router.add_delete("/nacos/v1/ns/instance", self._deregister)
        self.runner = None
        self.port = None

    async def start(self):
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.port = self.runner.addresses[0][1]

    async def stop(self):
        if self.runner:
            await self.runner.cleanup()

    def requests(self, method, path):
        return [c for c in self.calls if c[0] == method and c[1] == path]

    async def _login(self, request):
        form = await request.post()
        self.calls.append(("POST", "/nacos/v1/auth/login", dict(form)))
        if dict(form).get("username") == "test_user":
            return web.json_response({"accessToken": "tok-123"})
        return web.json_response({"error"}, status=401)

    async def _register(self, request):
        params = dict(request.query)
        self.calls.append(("POST", "/nacos/v1/ns/instance", params))
        if self.fail_next_register > 0:
            self.fail_next_register -= 1
            return web.Response(status=500, text="server is down")
        return web.Response(text="ok")

    async def _beat(self, request):
        params = dict(request.query)
        self.calls.append(("PUT", "/nacos/v1/ns/instance/beat", params))
        return web.json_response({"code": 10200, "message": "success"})

    async def _deregister(self, request):
        params = dict(request.query)
        self.calls.append(("DELETE", "/nacos/v1/ns/instance", params))
        return web.Response(text="ok")


class TestNacosRegistry(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fake = FakeNacosServer()
        await self.fake.start()

    async def asyncTearDown(self):
        await self.fake.stop()

    def _registry(self, **overrides) -> NacosRegistry:
        cfg = RegistryConfig(
            server_endpoint=f"127.0.0.1:{self.fake.port}",
            service_name="asrflow-test",
            service_host="10.1.2.3",
            namespace="public",
            group_name="DEFAULT_GROUP",
            username="test_user",
            password="test_pass_123",
            heartbeat_sec=0.05,
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return NacosRegistry(cfg, service_port=10195)

    async def test_register_beat_deregister_lifecycle(self):
        reg = self._registry()
        await reg.start()
        # login performed and token forwarded to the registration request
        logins = self.fake.requests("POST", "/nacos/v1/auth/login")
        self.assertEqual(len(logins), 1)
        self.assertEqual(logins[0][2]["username"], "test_user")

        registers = self.fake.requests("POST", "/nacos/v1/ns/instance")
        self.assertEqual(len(registers), 1)
        params = registers[0][2]
        self.assertEqual(params["serviceName"], "asrflow-test")
        self.assertEqual(params["ip"], "10.1.2.3")
        self.assertEqual(params["port"], "10195")
        self.assertEqual(params["namespaceId"], "public")
        self.assertEqual(params["groupName"], "DEFAULT_GROUP")
        self.assertEqual(params["accessToken"], "tok-123")
        self.assertEqual(params["ephemeral"], "true")

        # heartbeats flow while running
        await asyncio.sleep(0.25)
        beats = self.fake.requests("PUT", "/nacos/v1/ns/instance/beat")
        self.assertGreaterEqual(len(beats), 2)
        beat_payload = json.loads(beats[0][2]["beat"])
        self.assertEqual(beat_payload["ip"], "10.1.2.3")
        self.assertEqual(beat_payload["port"], 10195)

        # graceful stop deregisters the instance
        await reg.stop()
        deregisters = self.fake.requests("DELETE", "/nacos/v1/ns/instance")
        self.assertEqual(len(deregisters), 1)
        self.assertEqual(deregisters[0][2]["port"], "10195")

    async def test_retry_after_initial_register_failure(self):
        self.fake.fail_next_register = 1
        reg = self._registry()
        await reg.start()
        # first attempt failed, heartbeat loop retries and succeeds
        await asyncio.sleep(0.25)
        registers = self.fake.requests("POST", "/nacos/v1/ns/instance")
        self.assertGreaterEqual(len(registers), 2)
        self.assertTrue(reg._registered)
        await reg.stop()

    async def test_no_auth_when_username_empty(self):
        reg = self._registry(username="", password="")
        await reg.start()
        await asyncio.sleep(0.1)
        self.assertEqual(len(self.fake.requests("POST", "/nacos/v1/auth/login")), 0)
        registers = self.fake.requests("POST", "/nacos/v1/ns/instance")
        self.assertEqual(len(registers), 1)
        self.assertNotIn("accessToken", registers[0][2])
        await reg.stop()


class TestRegistryConfigEnvOverrides(unittest.TestCase):
    def test_env_overrides(self):
        env = {
            "NACOS_ENABLE": "true",
            "NACOS_SERVER_ENDPOINT": "10.9.8.7:8848",
            "NACOS_SERVICE_NAME": "asr-prod",
            "NACOS_SERVICE_HOST": "10.9.8.6",
            "NACOS_NAMESPACE": "prod-ns",
            "NACOS_GROUP_NAME": "ASR_GROUP",
            "NACOS_USERNAME": "alice",
            "NACOS_PASSWORD": "pw",
            "NACOS_HEARTBEAT_SEC": "7.5",
        }
        old = {k: os.environ.get(k) for k in env}
        try:
            os.environ.update(env)
            cfg = AppConfig.load(None)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.assertTrue(cfg.registry.enable)
        self.assertEqual(cfg.registry.server_endpoint, "10.9.8.7:8848")
        self.assertEqual(cfg.registry.service_name, "asr-prod")
        self.assertEqual(cfg.registry.service_host, "10.9.8.6")
        self.assertEqual(cfg.registry.namespace, "prod-ns")
        self.assertEqual(cfg.registry.group_name, "ASR_GROUP")
        self.assertEqual(cfg.registry.username, "alice")
        self.assertEqual(cfg.registry.password, "pw")
        self.assertEqual(cfg.registry.heartbeat_sec, 7.5)


if __name__ == "__main__":
    unittest.main()
