"""
Provider Presets & Factory — Provider 注册表 + 工厂
Maps to: src/services/api/ (provider resolution)

Known providers, their default endpoints, and the factory function.
已知 Provider 的默认端点和工厂函数。
"""
from __future__ import annotations

from typing import Optional

from claw_agent.providers.base import LLMProvider
from claw_agent.providers.openai_compat import OpenAICompatibleProvider
from claw_agent.providers.anthropic import AnthropicProvider
from claw_agent.providers.gemini import GeminiProvider


# ────────────────────────────────────────────────────────────────
# Provider registry — 注册表
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


# ────────────────────────────────────────────────────────────────
# Factory — 工厂函数
# ────────────────────────────────────────────────────────────────

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
