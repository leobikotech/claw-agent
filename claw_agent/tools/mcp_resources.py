"""
MCP Resource Tools — MCP 资源工具
Maps to: src/tools/ListMcpResourcesTool/ + src/tools/ReadMcpResourceTool/

Tools for browsing and reading resources exposed by MCP servers.
MCP 资源浏览和读取工具。
"""
from __future__ import annotations
import json
from typing import Any, Optional

from claw_agent.core.tool import Tool, ToolContext, RiskLevel


class ListMcpResourcesTool(Tool):
    """List resources from connected MCP servers / 列出 MCP 服务器资源
    Maps to: ListMcpResourcesTool in src/tools/ListMcpResourcesTool/

    Returns a JSON list of resources with uri, name, mimeType, description, and server.
    """
    name = "list_mcp_resources"
    description = (
        "List resources available from connected MCP servers. "
        "Resources provide read-only access to data like database schemas, "
        "API docs, or configuration. Optionally filter by server name."
    )
    risk_level = RiskLevel.LOW
    is_read_only = True
    is_destructive = False

    def __init__(self, mcp_manager: Any = None):
        """
        Args:
            mcp_manager: MCPManager instance. If None, tool returns empty.
        """
        self._mcp_manager = mcp_manager
        super().__init__()

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "server": {
                    "type": "string",
                    "description": (
                        "Optional server name to filter resources by. "
                        "If omitted, lists resources from all connected servers."
                    ),
                },
            },
        }

    async def call(self, arguments: dict[str, Any], context: ToolContext) -> str:
        if not self._mcp_manager:
            return "No MCP servers connected."

        server_name = arguments.get("server")
        resources = await self._mcp_manager.list_resources(server_name=server_name)

        if not resources:
            return (
                "No resources found. MCP servers may still provide tools "
                "even if they have no resources."
            )

        return json.dumps(resources, indent=2, ensure_ascii=False)


class ReadMcpResourceTool(Tool):
    """Read a specific resource from an MCP server / 读取 MCP 服务器资源
    Maps to: ReadMcpResourceTool in src/tools/ReadMcpResourceTool/

    Reads the content of a resource identified by its URI and server name.
    """
    name = "read_mcp_resource"
    description = (
        "Read the content of a resource from a connected MCP server. "
        "Use list_mcp_resources first to discover available resources and their URIs."
    )
    risk_level = RiskLevel.LOW
    is_read_only = True
    is_destructive = False

    def __init__(self, mcp_manager: Any = None):
        """
        Args:
            mcp_manager: MCPManager instance. If None, tool returns error.
        """
        self._mcp_manager = mcp_manager
        super().__init__()

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "server": {
                    "type": "string",
                    "description": "Name of the MCP server that provides the resource.",
                },
                "uri": {
                    "type": "string",
                    "description": "URI of the resource to read (from list_mcp_resources output).",
                },
            },
            "required": ["server", "uri"],
        }

    async def call(self, arguments: dict[str, Any], context: ToolContext) -> str:
        if not self._mcp_manager:
            return "Error: No MCP servers connected."

        server = arguments.get("server", "")
        uri = arguments.get("uri", "")

        if not server or not uri:
            return "Error: Both 'server' and 'uri' are required."

        try:
            content = await self._mcp_manager.read_resource(server, uri)
            return content
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error reading resource '{uri}' from '{server}': {e}"
