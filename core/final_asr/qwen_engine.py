import io
import re
import wave
import base64
import json
import asyncio
from typing import Dict, Any, List, Optional
import aiohttp
from loguru import logger

from core.final_asr.base import BaseFinalASREngine
from config.settings import FinalASRConfig


def pcm_to_wav_bytes(
    pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1, sampwidth: int = 2
) -> bytes:
    """Pack raw PCM16 bytes into standard WAV container."""
    if not pcm_bytes:
        return b""
    # Ensure 2-byte alignment
    if len(pcm_bytes) % 2 != 0:
        pcm_bytes = pcm_bytes[:-1]

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def clean_qwen_asr_text(text: str) -> str:
    """
    Clean up model outputs from vLLM online serving:
    - Strips special tokens: <|im_end|>, <|endoftext|>, <|zh|>, <|en|>
    - Strips timestamp markers: <|0.00|>, <|1.20|>
    - Strips HTML-like tags: <[^>]*>
    - Strips prompt echo: "语音转写：" prefix and (前文: ...)/(热词: ...)
      hints (also legacy (上下文: ...)/(提示词: ...)) echoed back by the model
    - Strips common language/task prefixes
    """
    if not text:
        return ""
    # Strip <|...|> and <...> tags
    cleaned = re.sub(r"<\|.*?\|>", "", text)
    cleaned = re.sub(r"<[^>]*>", "", cleaned)
    cleaned = re.sub(r"\[.*?\]", "", cleaned)
    # Strip prompt echo (vLLM 偶发把指令前缀/上下文提示原样回显进转写结果;
    # 兼容现用 (前文/热词) 与旧版 (上下文/提示词) 两种 prompt 措辞)
    cleaned = re.sub(r"语音转写[：:]\s*", "", cleaned)
    cleaned = re.sub(r"[（(](?:上下文|提示词|前文|热词)[：:].*?[）)]", "", cleaned)
    # Strip special artifacts
    cleaned = re.sub(r"/sil|endofbreak|FFFF", "", cleaned)
    # Strip language prefix like "zh:" or "中文:" if present at start
    cleaned = re.sub(r"^(zh|en|ja|yue|中文|英语|日语)[:：]\s*", "", cleaned, flags=re.IGNORECASE)
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


class Qwen3ASREngine(BaseFinalASREngine):
    """
    Qwen3-ASR Client communicating with vLLM Online Serving.
    Reference: https://github.com/QwenLM/Qwen3-ASR#deployment-with-vllm
    Supports both:
    1. /v1/chat/completions (OpenAI Chat Audio API with input_audio)
    2. /v1/audio/transcriptions (OpenAI Audio Transcriptions API)
    """

    def __init__(self, config: FinalASRConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None

        # Resolve URL endpoint
        url = self.config.vllm_url.rstrip("/")
        if not url.endswith("/v1/chat/completions") and not url.endswith("/v1/audio/transcriptions"):
            if "/v1" in url:
                url = f"{url}/chat/completions"
            else:
                url = f"{url}/v1/chat/completions"
        self._endpoint_url = url

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.hard_timeout_sec)
            headers = {"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}
            self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self._session

    @staticmethod
    def _sanitize_hotwords(hotwords: Optional[List[str]]) -> List[str]:
        if not hotwords:
            return []
        sanitized: List[str] = []
        for w in hotwords[:5]:
            if not isinstance(w, str):
                continue
            w = w.strip()[:30]
            if not w:
                continue
            # Allowlist: Chinese, alphanumeric and common punctuation
            if len(w) > 30:
                w = w[:30]
            sanitized.append(w)
        return sanitized

    def _build_prompt(self, hotwords: Optional[List[str]] = None, context: Optional[str] = None) -> str:
        prompt_text = self.config.prompt_prefix
        shw = self._sanitize_hotwords(hotwords)
        if shw:
            prompt_text += f" (热词: {', '.join(shw)})"
        limit = self.config.context_history_chars
        if context and limit > 0:
            ctx = context.strip()[:limit]
            if ctx:
                prompt_text += f" (前文: {ctx})"
        return prompt_text

    def _build_chat_payload(
        self,
        wav_b64: str,
        context: Optional[str] = None,
        hotwords: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Construct OpenAI-compatible chat completions payload for vLLM online serving.
        """
        prompt_text = self._build_prompt(hotwords=hotwords, context=context)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": wav_b64,
                            "format": "wav",
                        },
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        return {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }

    async def transcribe(
        self,
        audio_bytes: bytes,
        context: Optional[str] = None,
        hotwords: Optional[List[str]] = None,
    ) -> str:
        if not audio_bytes:
            return ""

        wav_data = pcm_to_wav_bytes(audio_bytes)
        session = await self._get_session()

        try:
            if self._endpoint_url.endswith("/transcriptions"):
                # Multipart /v1/audio/transcriptions API
                data = aiohttp.FormData()
                data.add_field(
                    "file",
                    wav_data,
                    filename="audio.wav",
                    content_type="audio/wav",
                )
                data.add_field("model", self.config.model_name)
                prompt_text = self._build_prompt(hotwords=hotwords, context=context)
                data.add_field("prompt", prompt_text)

                async with session.post(self._endpoint_url, data=data) as resp:
                    if resp.status != 200:
                        err_text = await resp.text()
                        logger.error(f"[Qwen3ASREngine] vLLM transcriptions failed {resp.status}: {err_text}")
                        raise RuntimeError(f"vLLM server error: status {resp.status}")
                    result_json = await resp.json()
                    raw_text = result_json.get("text", "")
                    return clean_qwen_asr_text(raw_text)
            else:
                # JSON /v1/chat/completions API (Default vLLM online serving)
                wav_b64 = base64.b64encode(wav_data).decode("utf-8")
                payload = self._build_chat_payload(wav_b64, context=context, hotwords=hotwords)

                async with session.post(self._endpoint_url, json=payload) as resp:
                    if resp.status != 200:
                        err_text = await resp.text()
                        logger.error(f"[Qwen3ASREngine] vLLM online serving failed {resp.status}: {err_text}")
                        raise RuntimeError(f"vLLM server error: status {resp.status}")

                    data = await resp.json()
                    choices = data.get("choices", [])
                    if choices and "message" in choices[0]:
                        raw_content = choices[0]["message"].get("content", "")
                        return clean_qwen_asr_text(raw_content)
                    return ""

        except asyncio.TimeoutError:
            logger.warning(
                f"[Qwen3ASREngine] Online serving request timed out after {self.config.hard_timeout_sec}s"
            )
            raise
        except Exception as e:
            logger.error(f"[Qwen3ASREngine] Inference failed: {e}")
            raise

    async def transcribe_batch(
        self, batch_items: List[Dict[str, Any]]
    ) -> List[Any]:
        tasks = [
            self.transcribe(
                item["audio_bytes"],
                context=item.get("context"),
                hotwords=item.get("hotwords"),
            )
            for item in batch_items
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
