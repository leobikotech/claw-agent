"""
Configuration — 配置系统
Maps to: bun:bundle feature(), src/utils/config.ts

Centralized config with provider-agnostic LLM settings.
集中配置，Provider 无关的 LLM 设置。
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Agent configuration / 智能体配置

    User only needs to set provider + api_key. Everything else has smart defaults.
    用户只需设置 provider + api_key，其他都有合理的默认值。
    """

    # --- LLM Provider (the key abstraction) ---
    provider: str = "minimax"  # openai | anthropic | gemini | minimax | kimi | deepseek | qwen
    api_key: str = field(default_factory=lambda: os.environ.get("MINIMAX_API_KEY", ""))
    base_url: Optional[str] = None   # Override default endpoint per provider
    model: Optional[str] = None      # None = use provider's default model

    # --- Agent behavior ---
    max_turns: int = 50           # Max tool-use loop iterations / 最大循环轮次
    max_tokens: int = 4096        # Max response tokens per turn
    temperature: float = 0.0
    cwd: str = field(default_factory=os.getcwd)  # Working directory / 工作目录

    # --- Permissions (maps to PermissionMode in Tool.ts) ---
    permission_mode: str = "default"  # default | auto | yolo

    # --- Memory ---
    memory_dir: Optional[str] = None  # None = auto (~/.claw/memory/)

    # --- Feature flags (maps to bun:bundle feature()) ---
    features: dict = field(default_factory=lambda: {
        "MEMORY": True,           # File-based memory system
        "DREAM": True,            # Background memory consolidation
        "COORDINATOR": False,     # Multi-agent orchestration
        "MCP": True,              # Model Context Protocol
        "SUB_AGENT": True,        # Sub-agent spawning
    })

    # --- MCP servers ---
    mcp_servers: list = field(default_factory=list)
    # Format: [{"name": "xxx", "command": "node", "args": ["/path/to/server.js"]}]

    # --- System prompt ---
    system_prompt: Optional[str] = None  # Custom override
    append_system_prompt: Optional[str] = None  # Appended to default

    def feature(self, name: str) -> bool:
        """Check if a feature is enabled / 检查功能是否启用"""
        return self.features.get(name, False)

    @property
    def effective_memory_dir(self) -> str:
        if self.memory_dir:
            return self.memory_dir
        return os.path.join(os.path.expanduser("~"), ".claw", "memory")

    @property
    def effective_model(self) -> str:
        """Resolve model name — use explicit or provider default / 解析模型名"""
        if self.model:
            return self.model
        from claw_agent.providers.presets import PROVIDER_PRESETS
        preset = PROVIDER_PRESETS.get(self.provider.lower(), {})
        return preset.get("model", "gpt-4o")

    @property
    def effective_api_key(self) -> str:
        """Resolve API key — explicit > env var per provider / 解析 API key"""
        if self.api_key:
            return self.api_key
        from claw_agent.providers.presets import PROVIDER_PRESETS
        preset = PROVIDER_PRESETS.get(self.provider.lower(), {})
        env_key = preset.get("env_key", "")
        return os.environ.get(env_key, "")
