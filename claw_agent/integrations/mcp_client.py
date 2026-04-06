"""
MCP Client — Model Context Protocol 客户端

Manages connections to MCP servers and bridges their tools into the agent.
管理 MCP 服务器连接，将其工具桥接到智能体系统中。

Supported transports / 支持的传输方式:
  - stdio  (default) — local process via stdin/stdout
  - sse    — Server-Sent Events over HTTP
  - http   — Streamable HTTP (MCP 2025-03-26 spec)
"""
from __future__ import annotations
import asyncio
import logging
import warnings
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from claw_agent.core.tool import Tool, ToolContext, ToolRegistry, RiskLevel

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────

# Cap on MCP tool descriptions sent to the model.
MAX_MCP_DESCRIPTION_LENGTH = 2048

# Connection retry config
MAX_CONNECT_RETRIES = 3
INITIAL_RETRY_BACKOFF_S = 1.0
MAX_RETRY_BACKOFF_S = 10.0
CONNECTION_TIMEOUT_S = 30.0


# ────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────

@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server / 单个 MCP 服务器配置

    Transport types:
      - stdio (default): local process — requires `command` + `args`
      - sse: Server-Sent Events — requires `url`
      - http: Streamable HTTP — requires `url`
    """
    name: str
    command: str = ""              # For stdio: e.g. "node", "python"
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    transport: str = "stdio"       # "stdio" | "sse" | "http"
    url: str = ""                  # For sse/http: server URL
    headers: dict[str, str] = field(default_factory=dict)  # For sse/http: custom headers


# ────────────────────────────────────────────────────────────────
# MCPBridgeTool — tool wrapping
# ────────────────────────────────────────────────────────────────

def _truncate_description(desc: str, max_len: int = MAX_MCP_DESCRIPTION_LENGTH) -> str:
    """Truncate MCP tool/server descriptions to prevent context overflow.
    """
    if len(desc) <= max_len:
        return desc
    return desc[:max_len] + "… [truncated]"


class MCPBridgeTool(Tool):
    """Bridges an MCP server tool into the agent's tool system / MCP 工具桥接
    """

    def __init__(self, server_name: str, tool_def: dict, session: ClientSession):
        self.name = f"mcp__{server_name}__{tool_def['name']}"
        self.description = _truncate_description(tool_def.get("description", ""))
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
                parts.append(getattr(block, "text"))
            else:
                parts.append(str(block))
        return "\n".join(parts) if parts else "(empty result)"


# ────────────────────────────────────────────────────────────────
# MCPManager — connection management
# ────────────────────────────────────────────────────────────────

class MCPManager:
    """Manages all MCP server connections / 管理所有 MCP 服务器连接
    Manages multiple MCP server connections with lifecycle management.

    Features:
      - Multiple transport support (stdio, sse, http)
      - Connection retry with exponential backoff
      - Server instructions extraction
      - Tool description truncation
    """

    def __init__(self):
        self._stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}
        self._server_instructions: dict[str, str] = {}

    async def connect(self, server: MCPServerConfig) -> ClientSession:
        """Connect to a single MCP server with retry / 连接到 MCP 服务器（带重试）
        """
        last_error: Optional[Exception] = None

        for attempt in range(MAX_CONNECT_RETRIES):
            try:
                return await self._connect_once(server)
            except Exception as e:
                last_error = e
                if attempt < MAX_CONNECT_RETRIES - 1:
                    backoff = min(
                        INITIAL_RETRY_BACKOFF_S * (2 ** attempt),
                        MAX_RETRY_BACKOFF_S,
                    )
                    logger.warning(
                        f"MCP connect attempt {attempt + 1}/{MAX_CONNECT_RETRIES} "
                        f"for '{server.name}' failed: {e}. Retrying in {backoff:.1f}s..."
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error(
                        f"Failed to connect to MCP server '{server.name}' "
                        f"after {MAX_CONNECT_RETRIES} attempts: {e}"
                    )

        raise last_error  # type: ignore[misc]

    async def _connect_once(self, server: MCPServerConfig) -> ClientSession:
        """Single connection attempt — dispatches by transport type.
        """
        transport_type = server.transport.lower()

        if transport_type == "stdio":
            return await self._connect_stdio(server)
        elif transport_type == "sse":
            return await self._connect_sse(server)
        elif transport_type == "http":
            return await self._connect_http(server)
        else:
            raise ValueError(
                f"Unsupported MCP transport: '{transport_type}'. "
                f"Supported: stdio, sse, http"
            )

    async def _connect_stdio(self, server: MCPServerConfig) -> ClientSession:
        """Connect via stdio transport / 通过 stdio 连接
        """
        logger.info(f"Connecting to MCP server via stdio: {server.name}")

        params = StdioServerParameters(
            command=server.command,
            args=server.args,
            env=server.env or None,
        )

        transport = await self._stack.enter_async_context(stdio_client(params))
        read, write = transport
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await asyncio.wait_for(session.initialize(), timeout=CONNECTION_TIMEOUT_S)

        self._sessions[server.name] = session
        self._extract_server_instructions(server.name, session)
        logger.info(f"Connected to MCP server via stdio: {server.name}")
        return session

    async def _connect_sse(self, server: MCPServerConfig) -> ClientSession:
        """Connect via SSE transport / 通过 SSE 连接
        """
        if not server.url:
            raise ValueError(f"MCP server '{server.name}' with transport='sse' requires a 'url'")

        logger.info(f"Connecting to MCP server via SSE: {server.name} → {server.url}")

        from mcp.client.sse import sse_client

        transport = await self._stack.enter_async_context(
            sse_client(server.url, headers=server.headers or None)
        )
        read, write = transport
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await asyncio.wait_for(session.initialize(), timeout=CONNECTION_TIMEOUT_S)

        self._sessions[server.name] = session
        self._extract_server_instructions(server.name, session)
        logger.info(f"Connected to MCP server via SSE: {server.name}")
        return session

    async def _connect_http(self, server: MCPServerConfig) -> ClientSession:
        """Connect via Streamable HTTP transport / 通过 HTTP 连接
        """
        if not server.url:
            raise ValueError(f"MCP server '{server.name}' with transport='http' requires a 'url'")

        logger.info(f"Connecting to MCP server via HTTP: {server.name} → {server.url}")

        from mcp.client.streamable_http import streamablehttp_client

        transport = await self._stack.enter_async_context(
            streamablehttp_client(server.url, headers=server.headers or None)
        )
        read, write, _ = transport
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await asyncio.wait_for(session.initialize(), timeout=CONNECTION_TIMEOUT_S)

        self._sessions[server.name] = session
        self._extract_server_instructions(server.name, session)
        logger.info(f"Connected to MCP server via HTTP: {server.name}")
        return session

    def _extract_server_instructions(self, name: str, session: ClientSession) -> None:
        """Extract and store server instructions after connection.
        """
        try:
            server_info = getattr(session, "server_info", None)
            raw = getattr(server_info, "instructions", None) if server_info else None
            if raw:
                self._server_instructions[name] = _truncate_description(raw)
                logger.info(
                    f"MCP server '{name}' instructions: "
                    f"{len(raw)} chars{' (truncated)' if len(raw) > MAX_MCP_DESCRIPTION_LENGTH else ''}"
                )
        except Exception as e:
            logger.debug(f"Could not extract instructions from '{name}': {e}")

    async def connect_all(self, servers: list[MCPServerConfig]) -> None:
        """Connect to all configured MCP servers / 连接所有 MCP 服务器"""
        for server in servers:
            try:
                await self.connect(server)
            except Exception as e:
                logger.error(f"Failed to connect MCP server '{server.name}': {e}")

    def discover_tools(self, registry: ToolRegistry) -> list[Tool]:
        """Discover and register tools from all connected MCP servers.

        .. deprecated::
            Use `discover_tools_async()` instead. This synchronous method
            cannot call the async MCP list_tools() API and always returns
            an empty list.
        """
        warnings.warn(
            "discover_tools() is deprecated — use discover_tools_async() instead. "
            "The synchronous version cannot call the async MCP API.",
            DeprecationWarning,
            stacklevel=2,
        )
        return []

    async def discover_tools_async(self, registry: ToolRegistry) -> list[Tool]:
        """Async version — discover and register all MCP tools / 异步发现并注册 MCP 工具
        """
        tools: list[Tool] = []
        for name, session in self._sessions.items():
            try:
                result = await session.list_tools()
                for td in result.tools:
                    raw_desc = td.description or ""
                    tool_def = {
                        "name": td.name,
                        "description": raw_desc,
                        "inputSchema": td.inputSchema if td.inputSchema else {"type": "object", "properties": {}},
                    }
                    bridge_tool = MCPBridgeTool(name, tool_def, session)
                    registry.register(bridge_tool)
                    tools.append(bridge_tool)
                    logger.info(f"Registered MCP tool: {bridge_tool.name}")
            except Exception as e:
                logger.error(f"Failed to discover tools from '{name}': {e}")
        return tools

    # ────────────────────────────────────────────────────────────
    # Resources — maps to ListMcpResourcesTool / ReadMcpResourceTool
    # ────────────────────────────────────────────────────────────

    async def list_resources(self, server_name: Optional[str] = None) -> list[dict]:
        """List resources from connected MCP servers.

        Args:
            server_name: Optional filter — only list resources from this server.

        Returns:
            List of resource dicts with keys: uri, name, mimeType, description, server
        """
        results: list[dict] = []
        targets = (
            {server_name: self._sessions[server_name]}
            if server_name and server_name in self._sessions
            else self._sessions
        )

        for name, session in targets.items():
            try:
                # Check if server supports resources capability
                server_info = getattr(session, "server_info", None)
                caps = getattr(server_info, "capabilities", None) if server_info else None
                if caps and not getattr(caps, 'resources', None):
                    continue

                resp = await session.list_resources()
                for r in resp.resources:
                    results.append({
                        "uri": str(r.uri),
                        "name": r.name,
                        "mimeType": getattr(r, "mimeType", None),
                        "description": getattr(r, "description", None),
                        "server": name,
                    })
            except Exception as e:
                logger.warning(f"Failed to list resources from '{name}': {e}")
        return results

    async def read_resource(self, server_name: str, uri: str) -> str:
        """Read a specific resource from an MCP server.

        Args:
            server_name: Name of the MCP server
            uri: Resource URI to read

        Returns:
            Resource content as string
        """
        session = self._sessions.get(server_name)
        if not session:
            raise ValueError(f"MCP server '{server_name}' not connected")

        result = await session.read_resource(uri)  # type: ignore[arg-type]
        parts = []
        for block in result.contents:
            if hasattr(block, "text"):
                parts.append(getattr(block, "text"))
            elif hasattr(block, "blob"):
                blob = getattr(block, "blob")
                parts.append(f"[binary data: {len(blob)} bytes]")
            else:
                parts.append(str(block))
        return "\n".join(parts) if parts else "(empty resource)"

    # ────────────────────────────────────────────────────────────
    # Server instructions — for PromptBuilder injection
    # ────────────────────────────────────────────────────────────

    def get_server_instructions(self) -> dict[str, str]:
        """Get all server instructions (already truncated).
        """
        return dict(self._server_instructions)

    def build_instructions_prompt(self) -> Optional[str]:
        """Build a prompt section from all server instructions.

        Returns:
            Formatted prompt string, or None if no servers have instructions.
        """
        if not self._server_instructions:
            return None

        parts = ["# MCP Server Instructions\n"]
        parts.append(
            "The following instructions were provided by connected MCP servers. "
            "Follow them when using the corresponding tools.\n"
        )
        for name, instructions in self._server_instructions.items():
            parts.append(f"## Server: {name}\n{instructions}\n")

        return "\n".join(parts)

    # ────────────────────────────────────────────────────────────
    # Lifecycle
    # ────────────────────────────────────────────────────────────

    def get_connected_servers(self) -> list[str]:
        """Get names of all connected servers."""
        return list(self._sessions.keys())

    async def close(self):
        """Close all connections / 关闭所有连接"""
        await self._stack.aclose()
        self._sessions.clear()
        self._server_instructions.clear()

