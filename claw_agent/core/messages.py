"""
Messages — 消息类型定义
Maps to: src/types/message.ts, src/Tool.ts (ToolResult)

Standardized message format compatible with OpenAI-style chat completions API.
标准化消息格式，兼容 OpenAI 风格的 Chat Completions API (MiniMax)。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import json


@dataclass
class UserMessage:
    """User input message / 用户输入消息"""
    content: str
    role: str = "user"

    def to_api(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class AssistantMessage:
    """LLM response message / LLM 响应消息"""
    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: Optional[str] = None  # MiniMax reasoning output
    role: str = "assistant"

    def to_api(self) -> dict:
        msg: dict[str, Any] = {"role": self.role}
        if self.content:
            msg["content"] = self.content
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_api() for tc in self.tool_calls]
        return msg


@dataclass
class SystemMessage:
    """System prompt message / 系统提示消息"""
    content: str
    role: str = "system"

    def to_api(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class ToolCall:
    """A single tool invocation from the LLM / LLM 发起的单次工具调用
    Maps to: ToolUseBlockParam in Anthropic SDK
    """
    id: str
    name: str
    arguments: dict[str, Any]

    def to_api(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass
class ToolResult:
    """Result returned to the LLM after tool execution / 工具执行后返回给 LLM 的结果
    Maps to: ToolResultBlockParam, ToolResult<T> in Tool.ts
    """
    tool_call_id: str
    content: str
    is_error: bool = False
    role: str = "tool"

    def to_api(self) -> dict:
        return {
            "role": self.role,
            "tool_call_id": self.tool_call_id,
            "content": self.content,
        }


# Type alias for message history / 消息历史类型别名
Message = UserMessage | AssistantMessage | SystemMessage | ToolResult


def messages_to_api(messages: list[Message]) -> list[dict]:
    """Convert message list to API format / 将消息列表转换为 API 格式"""
    return [m.to_api() for m in messages]
