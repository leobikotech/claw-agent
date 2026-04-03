"""
Claw Agent — Python agent framework inspired by Claude Code architecture.
Claw Agent — 基于 Claude Code 架构的 Python 智能体框架。

Core concepts / 核心概念:
  - Engine: The agent loop (provider-agnostic)
  - Provider: Unified LLM interface (OpenAI, Claude, Gemini, MiniMax, Kimi, DeepSeek)
  - Tool: Schema-driven, permission-gated tools
  - Memory: File-based persistent memory with dream consolidation
  - MCP: Model Context Protocol for external tool servers
"""
from claw_agent.core.engine import Engine
from claw_agent.core.tool import Tool, tool, ToolRegistry
from claw_agent.core.messages import UserMessage, AssistantMessage, ToolCall, ToolResult
from claw_agent.core.provider import (
    LLMProvider, LLMResponse, LLMToolCall,
    OpenAICompatibleProvider, AnthropicProvider, GeminiProvider,
    create_provider, PROVIDER_PRESETS,
)
from claw_agent.config import Config

__all__ = [
    "Engine", "Tool", "tool", "ToolRegistry", "Config",
    "UserMessage", "AssistantMessage", "ToolCall", "ToolResult",
    "LLMProvider", "LLMResponse", "LLMToolCall",
    "OpenAICompatibleProvider", "AnthropicProvider", "GeminiProvider",
    "create_provider", "PROVIDER_PRESETS",
]
