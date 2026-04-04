"""
Tool — 工具基类 + 注册器 + 装饰器
Maps to: src/Tool.ts (Tool<Input,Output>, buildTool(), ToolDef)

Core abstraction: schema-driven, permission-gated tools with a registry.
核心抽象：Schema 驱动、权限门控的工具系统，配合注册器。
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from claw_agent.config import Config


# --- Risk levels (maps to BashTool security classification) ---
class RiskLevel(str, Enum):
    LOW = "low"         # Read-only operations / 只读操作
    MEDIUM = "medium"   # File writes, non-destructive / 文件写入
    HIGH = "high"       # Destructive: delete, network, shell / 破坏性操作


# --- Permission result (maps to PermissionResult in types/permissions.ts) ---
@dataclass
class PermissionCheck:
    allowed: bool
    reason: str = ""


class Tool(ABC):
    """Base class for all tools / 所有工具的基类
    Maps to: Tool<Input, Output> in src/Tool.ts
    """
    name: str = ""
    description: str = ""
    risk_level: RiskLevel = RiskLevel.LOW
    is_read_only: bool = True
    is_destructive: bool = False

    @abstractmethod
    def get_parameters(self) -> dict:
        """JSON Schema for input parameters / 输入参数的 JSON Schema
        Maps to: inputSchema (Zod → JSON Schema)
        """
        ...

    @abstractmethod
    async def call(self, arguments: dict[str, Any], context: ToolContext) -> str:
        """Execute the tool / 执行工具
        Maps to: tool.call(args, context, canUseTool, ...) in Tool.ts
        """
        ...

    def check_permissions(self, arguments: dict[str, Any], config: Config) -> PermissionCheck:
        """Check if this tool use is allowed / 检查工具使用权限
        Maps to: checkPermissions() in Tool.ts
        """
        return PermissionCheck(allowed=True)

    def to_api(self) -> dict:
        """Convert to OpenAI-compatible tool definition / 转为 API 工具定义"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_parameters(),
            },
        }


@dataclass
class ToolContext:
    """Runtime context passed to tool.call() / 传递给 tool.call() 的运行时上下文
    Maps to: ToolUseContext in src/Tool.ts
    """
    cwd: str = "."
    config: Optional[Config] = None
    messages: list = field(default_factory=list)
    # Agent nesting depth (0 = main thread) / 代理嵌套深度
    depth: int = 0
    # Abort signal (static snapshot — prefer is_aborted property for live check)
    aborted: bool = False
    # Back-reference to Engine for background task registration (streaming re-entry)
    # 引擎反向引用，用于后台任务注册（流式重入架构）
    engine: Optional[Any] = None

    @property
    def is_aborted(self) -> bool:
        """Live abort check — delegates to engine if available.
        Prefer this over the static `aborted` field, which is a snapshot
        that won't reflect an abort signal received after context creation.
        """
        if self.engine is not None:
            return self.engine.aborted
        return self.aborted


# --- Tool Registry (maps to getAllBaseTools() + getTools() in tools.ts) ---
class ToolRegistry:
    """Global tool registry / 全局工具注册器
    Maps to: tools.ts getAllBaseTools(), assembleToolPool()
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool_instance: Tool) -> Tool:
        """Register a tool instance / 注册工具实例"""
        self._tools[tool_instance.name] = tool_instance
        return tool_instance

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def to_api(self) -> list[dict]:
        """All tools as API definitions / 所有工具的 API 定义"""
        return [t.to_api() for t in self._tools.values()]

    def find(self, name: str) -> Optional[Tool]:
        """Find tool by name or alias / 按名称查找工具
        Maps to: findToolByName() in Tool.ts
        """
        return self._tools.get(name)


# Default global registry / 默认全局注册器
_default_registry = ToolRegistry()


# --- @tool decorator (maps to buildTool() in Tool.ts) ---
def tool(
    name: str,
    description: str = "",
    parameters: Optional[dict] = None,
    risk_level: RiskLevel = RiskLevel.LOW,
    is_read_only: bool = True,
    registry: Optional[ToolRegistry] = None,
):
    """Decorator to create a tool from a function / 装饰器：将函数转为工具
    Maps to: buildTool(def) in Tool.ts — fills defaults, registers

    Usage / 用法:
        @tool("my_tool", description="Does something", parameters={...})
        async def my_tool(args, ctx):
            return "result"
    """
    reg = registry or _default_registry

    def decorator(fn: Callable) -> Tool:
        class FnTool(Tool):
            pass

        instance = FnTool()
        instance.name = name
        instance.description = description or fn.__doc__ or ""
        instance.risk_level = risk_level
        instance.is_read_only = is_read_only
        instance._parameters = parameters or {"type": "object", "properties": {}}
        instance._fn = fn  # type: ignore

        def get_parameters(self_inner) -> dict:
            return self_inner._parameters

        async def call_fn(self_inner, arguments: dict[str, Any], context: ToolContext) -> str:
            return await self_inner._fn(arguments, context)

        instance.get_parameters = get_parameters.__get__(instance)  # type: ignore
        instance.call = call_fn.__get__(instance)  # type: ignore

        reg.register(instance)
        return instance

    return decorator
