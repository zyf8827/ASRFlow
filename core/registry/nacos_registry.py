import json
import asyncio
import time
from typing import Optional
import aiohttp
from loguru import logger

from config.settings import RegistryConfig


class NacosRegistry:
    """
    Registers the ASRFlow WS service into Nacos as an ephemeral instance
    (heartbeat kept alive) using the Nacos Open API over HTTP.

    Only depends on aiohttp (already a project dependency) — no SDK, so the
    deliberately minimal requirements.txt stays untouched.

    Registration is best-effort: failures are logged and retried on the next
    heartbeat / re-registration round; they never crash the ASR service.
    """

    def __init__(self, config: RegistryConfig, service_port: int):
        self.config = config
        self.service_port = service_port
        self._session: Optional[aiohttp.ClientSession] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._access_token: Optional[str] = None
        # token 签发时刻与服务端 TTL(秒), 供 0.8xTTL 主动续期 (无 TTL 信息时不续期)
        self._token_ttl_sec: Optional[float] = None
        self._token_acquired_at: float = 0.0
        self._registered = False

    # ------------------------------------------------------------------ utils
    @property
    def _base_url(self) -> str:
        return f"http://{self.config.server_endpoint.rstrip('/')}"

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10.0))
        return self._session

    async def _login(self) -> bool:
        """Fetch an accessToken when username/password auth is enabled."""
        if not self.config.username:
            return True
        session = await self._get_session()
        try:
            async with session.post(
                f"{self._base_url}/nacos/v1/auth/login",
                data={"username": self.config.username, "password": self.config.password},
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(f"[NacosRegistry] Login failed ({resp.status}): {body[:200]}")
                    return False
                data = await resp.json(content_type=None)
                self._access_token = data.get("accessToken")
                try:
                    ttl = float(data.get("tokenTtl") or 0)
                except (TypeError, ValueError):
                    ttl = 0.0
                self._token_ttl_sec = ttl or None
                self._token_acquired_at = time.monotonic()
                return True
        except Exception as e:
            logger.error(f"[NacosRegistry] Login request failed: {e}")
            return False

    def _auth_params(self) -> dict:
        return {"accessToken": self._access_token} if self._access_token else {}

    async def _refresh_token_if_stale(self):
        """token 用到 0.8xTTL 即主动重新登录 (Nacos 默认 tokenTtl=5h)。

        纯被动刷新下, token 到期后首个心跳必然 403 一次才触发恢复; 主动
        续期消除该周期性告警。登录失败仅记日志, 心跳失败时的被动恢复
        (re-login + re-register) 仍兜底。无 TTL 信息 (旧版 Nacos) 时跳过。
        """
        if not self._access_token or not self._token_ttl_sec:
            return
        age = time.monotonic() - self._token_acquired_at
        if age < self._token_ttl_sec * 0.8:
            return
        logger.info(
            f"[NacosRegistry] Refreshing access token proactively "
            f"(age {age:.0f}s >= 0.8*TTL {self._token_ttl_sec:.0f}s)"
        )
        await self._login()

    def _instance_params(self) -> dict:
        c = self.config
        return {
            "serviceName": c.service_name,
            "groupName": c.group_name,
            "namespaceId": c.namespace,
            "clusterName": c.cluster_name,
            "ip": c.service_host,
            "port": self.service_port,
            "ephemeral": "true",
            "weight": 1.0,
            "healthy": "true",
            **self._auth_params(),
        }

    async def _register(self) -> bool:
        session = await self._get_session()
        try:
            async with session.post(
                f"{self._base_url}/nacos/v1/ns/instance", params=self._instance_params()
            ) as resp:
                body = await resp.text()
                if resp.status == 200 and "ok" in body.lower():
                    self._registered = True
                    return True
                # Token expiry: re-login once and retry
                if resp.status in (401, 403):
                    if await self._login():
                        async with session.post(
                            f"{self._base_url}/nacos/v1/ns/instance",
                            params=self._instance_params(),
                        ) as retry:
                            if retry.status == 200:
                                self._registered = True
                                return True
                logger.error(f"[NacosRegistry] Register failed ({resp.status}): {body[:200]}")
        except Exception as e:
            logger.error(f"[NacosRegistry] Register request failed: {e}")
        return False

    async def _send_beat(self) -> bool:
        c = self.config
        beat = {
            "serviceName": c.service_name,
            "ip": c.service_host,
            "port": self.service_port,
            "cluster": c.cluster_name,
            "weight": 1.0,
        }
        params = {
            "serviceName": c.service_name,
            "groupName": c.group_name,
            "namespaceId": c.namespace,
            "ip": c.service_host,
            "port": self.service_port,
            "beat": json.dumps(beat, ensure_ascii=False),
            **self._auth_params(),
        }
        session = await self._get_session()
        try:
            async with session.put(f"{self._base_url}/nacos/v1/ns/instance/beat", params=params) as resp:
                if resp.status == 200:
                    return True
                body = await resp.text()
                logger.warning(f"[NacosRegistry] Heartbeat failed ({resp.status}): {body[:200]}")
        except Exception as e:
            logger.warning(f"[NacosRegistry] Heartbeat request failed: {e}")
        return False

    async def _heartbeat_loop(self):
        while True:
            await asyncio.sleep(self.config.heartbeat_sec)
            await self._refresh_token_if_stale()
            if not self._registered:
                # Initial registration failed (e.g. Nacos briefly down):
                # keep retrying instead of dying
                await self._register()
                if self._registered:
                    logger.info(
                        f"[NacosRegistry] (re)registered {self.config.service_name} "
                        f"at {self.config.service_host}:{self.service_port}"
                    )
                continue
            if not await self._send_beat():
                # Beat failing (expired token / restarted Nacos): refresh and
                # re-register to guarantee the instance is present
                await self._login()
                if await self._register():
                    logger.info(
                        "[NacosRegistry] Recovered after failed heartbeat "
                        "(re-login + re-register)"
                    )

    # ----------------------------------------------------------------- public
    async def start(self):
        if await self._login():
            ok = await self._register()
            level = logger.info if ok else logger.error
            level(
                f"[NacosRegistry] {'Registered' if ok else 'Failed to register'} "
                f"{self.config.service_name} -> nacos://{self.config.server_endpoint}"
                f"/{self.config.namespace}/{self.config.group_name} "
                f"instance={self.config.service_host}:{self.service_port}"
            )
        else:
            logger.error(
                "[NacosRegistry] Nacos login failed; will keep retrying in heartbeat loop"
            )
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info(
            f"[NacosRegistry] Heartbeat started (interval {self.config.heartbeat_sec}s)"
        )

    async def stop(self):
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        if self._registered:
            session = await self._get_session()
            try:
                async with session.delete(
                    f"{self._base_url}/nacos/v1/ns/instance", params=self._instance_params()
                ) as resp:
                    logger.info(f"[NacosRegistry] Deregistered ({resp.status})")
            except Exception as e:
                logger.warning(f"[NacosRegistry] Deregister failed: {e}")
            self._registered = False

        if self._session and not self._session.closed:
            await self._session.close()
