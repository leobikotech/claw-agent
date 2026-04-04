"""
Providers — LLM Provider 统一接入层
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
from claw_agent.providers.base import LLMProvider, LLMResponse, LLMToolCall
from claw_agent.providers.openai_compat import OpenAICompatibleProvider
from claw_agent.providers.anthropic import AnthropicProvider
from claw_agent.providers.gemini import GeminiProvider
from claw_agent.providers.presets import (
    PROVIDER_PRESETS, create_provider, auto_detect_provider,
)
from claw_agent.providers.tokens import (
    estimate_tokens_text, estimate_tokens_messages, get_context_window,
)

__all__ = [
    "LLMProvider", "LLMResponse", "LLMToolCall",
    "OpenAICompatibleProvider", "AnthropicProvider", "GeminiProvider",
    "PROVIDER_PRESETS", "create_provider", "auto_detect_provider",
    "estimate_tokens_text", "estimate_tokens_messages", "get_context_window",
]
