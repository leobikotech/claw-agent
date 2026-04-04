"""
LLM Provider — 大模型统一接入中间层
Unified abstraction layer for all LLM providers.
统一的 LLM Provider 抽象层，让用户只需提供 API key 即可切换模型。

Supported providers / 支持的供应商:
  - OpenAI     (gpt-4o, gpt-4.1, ...)
  - Anthropic  (claude-sonnet-4-20250514, ...)
  - Google     (gemini-2.5-flash, ...)
  - MiniMax    (MiniMax-M2.7, ...)
  - Moonshot   (kimi-k2.5, ...)
  - DeepSeek   (deepseek-chat, deepseek-reasoner, ...)
  - Any OpenAI-compatible API

Architecture / 架构:
  Engine → LLMProvider.chat() → [OpenAI|Anthropic|Gemini] SDK → response
  Engine 只调用 provider.chat()，不关心底层用的是哪家 API。
"""
from __future__ import annotations
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# Unified response — 统一响应格式
# ────────────────────────────────────────────────────────────────

@dataclass
class LLMToolCall:
    """Unified tool call / 统一工具调用"""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Unified LLM response / 统一 LLM 响应
    All providers normalize their responses into this format.
    所有 Provider 将其响应归一化为此格式。
    """
    content: Optional[str] = None
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    thinking: Optional[str] = None        # Reasoning/thinking content
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)  # tokens


# ────────────────────────────────────────────────────────────────
# Base provider — Provider 基类
# ────────────────────────────────────────────────────────────────

class LLMProvider(ABC):
    """Base class for all LLM providers / 所有 LLM Provider 的基类"""

    name: str = "base"

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Send a chat completion request / 发送聊天补全请求

        Args:
            messages: Normalized message list (OpenAI format as baseline)
            tools: Tool definitions in OpenAI format
            **kwargs: model, temperature, max_tokens, etc.
        Returns:
            LLMResponse with normalized content and tool_calls
        """
        ...


# ────────────────────────────────────────────────────────────────
# OpenAI-compatible provider (covers OpenAI, MiniMax, Kimi, DeepSeek, etc.)
# ────────────────────────────────────────────────────────────────

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


# ────────────────────────────────────────────────────────────────
# Anthropic provider (Claude)
# ────────────────────────────────────────────────────────────────

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


# ────────────────────────────────────────────────────────────────
# Google Gemini provider
# ────────────────────────────────────────────────────────────────

class GeminiProvider(LLMProvider):
    """Provider for Google Gemini / Gemini API Provider
    Uses the google-genai SDK with manual tool handling.
    使用 google-genai SDK，手动处理工具调用。
    """

    name = "gemini"

    def __init__(self, api_key: str, **kwargs):
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._genai = genai

    async def chat(self, messages: list[dict], tools: list[dict] | None = None, **kwargs) -> LLMResponse:
        from google.genai import types
        import asyncio

        model = kwargs.get("model", "gemini-2.5-flash")

        # Convert OpenAI messages → Gemini contents
        contents = []
        system_instruction = None
        for m in messages:
            if m["role"] == "system":
                system_instruction = m["content"]
            elif m["role"] == "user":
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=m["content"])]))
            elif m["role"] == "assistant":
                parts = []
                if m.get("content"):
                    parts.append(types.Part.from_text(text=m["content"]))
                if m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        fn = tc["function"] if isinstance(tc.get("function"), dict) else tc
                        args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                        parts.append(types.Part.from_function_call(name=fn["name"], args=args))
                contents.append(types.Content(role="model", parts=parts))
            elif m["role"] == "tool":
                # Look up the function name from the tool_call_id by scanning
                # preceding assistant messages for matching tool calls
                fn_name = "tool"  # fallback
                tool_call_id = m.get("tool_call_id", "")
                for prev in reversed(messages[:messages.index(m)]):
                    if prev.get("role") == "assistant" and prev.get("tool_calls"):
                        for tc in prev["tool_calls"]:
                            if tc.get("id") == tool_call_id:
                                fn = tc.get("function", tc)
                                fn_name = fn.get("name", "tool") if isinstance(fn, dict) else "tool"
                                break
                        break
                parts = [types.Part.from_function_response(
                    name=fn_name,
                    response={"result": m["content"]},
                )]
                contents.append(types.Content(role="user", parts=parts))

        # Convert OpenAI tools → Gemini tool declarations
        gemini_tools = None
        if tools:
            declarations = []
            for t in tools:
                fn = t["function"]
                declarations.append(types.FunctionDeclaration(
                    name=fn["name"],
                    description=fn.get("description", ""),
                    parameters=fn.get("parameters"),
                ))
            gemini_tools = [types.Tool(function_declarations=declarations)]

        config = types.GenerateContentConfig(
            temperature=kwargs.get("temperature", 0.0),
            max_output_tokens=kwargs.get("max_tokens", 4096),
            tools=gemini_tools,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        # Gemini SDK is sync by default — run in executor
        response = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self._client.models.generate_content(
                model=model, contents=contents, config=config,
            ),
        )

        # Normalize
        content_text = ""
        tool_calls = []
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.text:
                    content_text += part.text
                elif part.function_call:
                    fc = part.function_call
                    tool_calls.append(LLMToolCall(
                        id=f"gemini_{fc.name}_{id(fc)}",
                        name=fc.name,
                        arguments=dict(fc.args) if fc.args else {},
                    ))

        return LLMResponse(
            content=content_text or None,
            tool_calls=tool_calls,
            finish_reason="stop",
            usage={},
        )


# ────────────────────────────────────────────────────────────────
# Provider registry + factory — 注册表 + 工厂
# ────────────────────────────────────────────────────────────────

# Known providers with their default base_url and models
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    # OpenAI
    "openai":    {"base_url": "https://api.openai.com/v1",        "model": "gpt-4o",              "env_key": "OPENAI_API_KEY"},
    # Anthropic
    "anthropic": {"base_url": "",                                  "model": "claude-sonnet-4-20250514",    "env_key": "ANTHROPIC_API_KEY"},
    # Google Gemini
    "gemini":    {"base_url": "",                                  "model": "gemini-2.5-flash",    "env_key": "GEMINI_API_KEY"},
    # MiniMax
    "minimax":   {"base_url": "https://api.minimaxi.com/v1",      "model": "MiniMax-M2.7",        "env_key": "MINIMAX_API_KEY"},
    # Moonshot / Kimi
    "kimi":      {"base_url": "https://api.moonshot.cn/v1",        "model": "kimi-k2.5",           "env_key": "KIMI_API_KEY"},
    "moonshot":  {"base_url": "https://api.moonshot.cn/v1",        "model": "kimi-k2.5",           "env_key": "MOONSHOT_API_KEY"},
    # DeepSeek
    "deepseek":  {"base_url": "https://api.deepseek.com/v1",      "model": "deepseek-chat",       "env_key": "DEEPSEEK_API_KEY"},
    # Qwen (Alibaba)
    "qwen":      {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus", "env_key": "QWEN_API_KEY"},
}


def create_provider(
    provider: str,
    api_key: str,
    base_url: Optional[str] = None,
    **kwargs,
) -> LLMProvider:
    """Factory function to create a provider / 工厂函数创建 Provider

    Usage / 用法:
        provider = create_provider("minimax", api_key="sk-...")
        provider = create_provider("anthropic", api_key="sk-ant-...")
        provider = create_provider("openai", api_key="sk-...", base_url="https://custom.api/v1")

    User only needs to specify provider name + API key.
    用户只需指定 provider 名称 + API key。
    """
    provider_lower = provider.lower()
    preset = PROVIDER_PRESETS.get(provider_lower, {})

    # Anthropic — uses its own SDK
    if provider_lower == "anthropic":
        return AnthropicProvider(api_key=api_key, **kwargs)

    # Gemini — uses google-genai SDK
    if provider_lower == "gemini":
        return GeminiProvider(api_key=api_key, **kwargs)

    # Everything else → OpenAI-compatible
    resolved_url = base_url or preset.get("base_url", "https://api.openai.com/v1")
    return OpenAICompatibleProvider(api_key=api_key, base_url=resolved_url, **kwargs)


def auto_detect_provider(api_key: str) -> tuple[str, str]:
    """Auto-detect provider from API key format / 根据 API key 格式自动检测 Provider

    Returns (provider_name, base_url)
    """
    if api_key.startswith("sk-ant-"):
        return "anthropic", ""
    if api_key.startswith("AIza"):
        return "gemini", ""
    # For OpenAI-compatible keys, default to MiniMax as per user preference
    return "minimax", PROVIDER_PRESETS["minimax"]["base_url"]
