"""
OpenAI Compatible Provider — OpenAI 兼容 API
Maps to: src/services/api/ (OpenAI path)

Covers: OpenAI, MiniMax, Moonshot/Kimi, DeepSeek, Qwen, and any custom endpoint.
涵盖：OpenAI、MiniMax、Kimi、DeepSeek、Qwen 及任何自定义端点。
"""
from __future__ import annotations

import json
import logging
from typing import Any

from claw_agent.providers.base import LLMProvider, LLMResponse, LLMToolCall

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """Provider for any OpenAI-compatible API / OpenAI 兼容 API 的 Provider
    Covers: OpenAI, MiniMax, Moonshot/Kimi, DeepSeek, and any custom endpoint.
    涵盖：OpenAI、MiniMax、Kimi、DeepSeek 及任何自定义端点。
    """

    name = "openai_compatible"

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", **kwargs):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._extra = kwargs  # e.g. reasoning_split for MiniMax

    async def chat(self, messages: list[dict], tools: list[dict] | None = None, **kwargs) -> LLMResponse:
        req: dict[str, Any] = {
            "model": kwargs.get("model", "gpt-4o"),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.0),
        }
        if tools:
            req["tools"] = tools
            req["tool_choice"] = kwargs.get("tool_choice", "auto")

        # Pass through any extra params (e.g. reasoning_split for MiniMax)
        for k, v in self._extra.items():
            req[k] = v

        response = await self._client.chat.completions.create(**req)
        choice = response.choices[0]
        msg = choice.message

        # Normalize tool calls
        tool_calls = []
        for tc in (msg.tool_calls or []):
            tool_calls.append(LLMToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments),
            ))

        # Extract thinking/reasoning if present (MiniMax, DeepSeek)
        thinking = None
        if hasattr(msg, "reasoning_content") and msg.reasoning_content:
            thinking = msg.reasoning_content

        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            thinking=thinking,
            finish_reason=choice.finish_reason or "stop",
            usage={
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
            } if response.usage else {},
        )
