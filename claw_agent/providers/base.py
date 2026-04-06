"""
Provider Base — LLM Provider 基类与统一响应格式

Core abstractions that all providers share.
所有 Provider 共享的核心抽象。
"""
from __future__ import annotations

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
