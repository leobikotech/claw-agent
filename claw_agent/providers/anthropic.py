"""

Handles the Anthropic-specific message format and tool_use blocks.
处理 Anthropic 特有的消息格式和 tool_use 内容块。
"""
from __future__ import annotations

import json
import logging
from typing import Any

from claw_agent.providers.base import LLMProvider, LLMResponse, LLMToolCall

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Provider for Anthropic Claude / Claude API Provider
    Handles the Anthropic-specific message format and tool_use blocks.
    处理 Anthropic 特有的消息格式和 tool_use 内容块。
    """

    name = "anthropic"

    def __init__(self, api_key: str, **kwargs):
        import anthropic
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def chat(self, messages: list[dict], tools: list[dict] | None = None, **kwargs) -> LLMResponse:
        # Convert OpenAI-format messages → Anthropic format
        system_text = ""
        anthropic_messages = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            elif m["role"] == "tool":
                # Anthropic expects tool results inside user messages
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m["tool_call_id"],
                        "content": m["content"],
                    }],
                })
            elif m["role"] == "assistant" and m.get("tool_calls"):
                # Convert assistant tool_calls to Anthropic content blocks
                content_blocks = []
                if m.get("content"):
                    content_blocks.append({"type": "text", "text": m["content"]})
                for tc in m["tool_calls"]:
                    fn = tc["function"] if isinstance(tc.get("function"), dict) else tc
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": fn["name"],
                        "input": json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"],
                    })
                anthropic_messages.append({"role": "assistant", "content": content_blocks})
            else:
                anthropic_messages.append({"role": m["role"], "content": m["content"]})

        # Convert OpenAI-format tools → Anthropic format
        anthropic_tools = []
        if tools:
            for t in tools:
                fn = t["function"]
                anthropic_tools.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                })

        req: dict[str, Any] = {
            "model": kwargs.get("model", "claude-sonnet-4-20250514"),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "messages": anthropic_messages,
        }
        if system_text:
            req["system"] = system_text
        if anthropic_tools:
            req["tools"] = anthropic_tools

        response = await self._client.messages.create(**req)

        # Normalize response
        content_text = ""
        tool_calls = []
        thinking = None

        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "thinking":
                thinking = block.thinking
            elif block.type == "tool_use":
                tool_calls.append(LLMToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        return LLMResponse(
            content=content_text or None,
            tool_calls=tool_calls,
            thinking=thinking,
            finish_reason=response.stop_reason or "stop",
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            },
        )
