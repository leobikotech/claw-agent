"""
MCP Client — Model Context Protocol 客户端
Maps to: src/services/mcp/ (MCPServerConnection, types.ts)

Manages connections to MCP servers and bridges their tools into the agent.
管理 MCP 服务器连接，将其工具桥接到智能体系统中。
"""
from __future__ import annotations
import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from claw_agent.core.tool import Tool, ToolContext, ToolRegistry, RiskLevel

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server / 单个 MCP 服务器配置"""
    name: str
    command: str            # e.g. "node", "python"
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


class MCPBridgeTool(Tool):
    """Bridges an MCP server tool into the agent's tool system / MCP 工具桥接
    Maps to: MCPTool in src/tools/MCPTool/MCPTool.ts
    """

    def __init__(self, server_name: str, tool_def: dict, session: ClientSession):
        self.name = f"mcp__{server_name}__{tool_def['name']}"
        self.description = tool_def.get("description", "")
        self._raw_name = tool_def["name"]
        self._schema = tool_def.get("inputSchema", {"type": "object", "properties": {}})
        self._session = session
        self.risk_level = RiskLevel.MEDIUM  # MCP tools are external
        self.is_read_only = False
        self.is_destructive = False

    def get_parameters(self) -> dict:
        return self._schema

    async def call(self, arguments: dict[str, Any], context: ToolContext) -> str:
        """Delegate execution to the MCP server / 委托 MCP 服务器执行"""
        result = await self._session.call_tool(self._raw_name, arguments=arguments)
        # Extract text content from MCP result
        parts = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            else:
                parts.append(str(block))
        return "\n".join(parts) if parts else "(empty result)"


class MCPManager:
    """Manages all MCP server connections / 管理所有 MCP 服务器连接
    Maps to: MCPServerConnection[] management in Claude Code
    """

    def __init__(self):
        self._stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}

    async def connect(self, server: MCPServerConfig) -> ClientSession:
        """Connect to a single MCP server / 连接到 MCP 服务器"""
        logger.info(f"Connecting to MCP server: {server.name}")

        params = StdioServerParameters(
            command=server.command,
            args=server.args,
            env=server.env or None,
        )

        transport = await self._stack.enter_async_context(stdio_client(params))
        read, write = transport
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        self._sessions[server.name] = session
        logger.info(f"Connected to MCP server: {server.name}")
        return session

    async def connect_all(self, servers: list[MCPServerConfig]) -> None:
        """Connect to all configured MCP servers / 连接所有 MCP 服务器"""
        for server in servers:
            try:
                await self.connect(server)
            except Exception as e:
                logger.error(f"Failed to connect MCP server '{server.name}': {e}")

    def discover_tools(self, registry: ToolRegistry) -> list[Tool]:
        """Discover and register tools from all connected MCP servers / 发现并注册 MCP 工具
        Maps to: assembleToolPool() MCP tool portion in tools.ts
        """
        tools: list[Tool] = []
        for name, session in self._sessions.items():
            # list_tools is async but we need it synchronous here for registration
            # The caller should use discover_tools_async instead
            pass
        return tools

    async def discover_tools_async(self, registry: ToolRegistry) -> list[Tool]:
        """Async version — discover and register all MCP tools / 异步发现并注册 MCP 工具"""
        tools: list[Tool] = []
        for name, session in self._sessions.items():
            try:
                result = await session.list_tools()
                for td in result.tools:
                    tool_def = {
                        "name": td.name,
                        "description": td.description or "",
                        "inputSchema": td.inputSchema if td.inputSchema else {"type": "object", "properties": {}},
                    }
                    bridge_tool = MCPBridgeTool(name, tool_def, session)
                    registry.register(bridge_tool)
                    tools.append(bridge_tool)
                    logger.info(f"Registered MCP tool: {bridge_tool.name}")
            except Exception as e:
                logger.error(f"Failed to discover tools from '{name}': {e}")
        return tools

    async def close(self):
        """Close all connections / 关闭所有连接"""
        await self._stack.aclose()
        self._sessions.clear()
