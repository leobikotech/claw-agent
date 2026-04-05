"""
MCP Example — Model Context Protocol 示例
Connect to MCP servers and use their tools + resources in the agent.
连接 MCP 服务器并在智能体中使用其工具与资源。

Supported transports: stdio (default), sse, http
"""
import asyncio
from claw_agent import Engine, Config
from claw_agent.tools import get_default_tools
from claw_agent.integrations.mcp_client import MCPManager, MCPServerConfig
from claw_agent.tools.mcp_resources import ListMcpResourcesTool, ReadMcpResourceTool


async def main():
    config = Config()

    # --- Define MCP servers to connect ---
    servers = [
        # Example 1: stdio transport (default) — local MCP server process
        MCPServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        ),
        # Example 2: SSE transport — remote MCP server via Server-Sent Events
        # MCPServerConfig(
        #     name="remote_db",
        #     transport="sse",
        #     url="http://localhost:3001/sse",
        #     headers={"Authorization": "Bearer <token>"},
        # ),
        # Example 3: HTTP transport — Streamable HTTP (MCP 2025-03-26 spec)
        # MCPServerConfig(
        #     name="api_server",
        #     transport="http",
        #     url="http://localhost:3002/mcp",
        # ),
    ]

    # --- Connect to MCP servers (with automatic retry) ---
    mcp = MCPManager()
    await mcp.connect_all(servers)

    # --- Check for server instructions ---
    instructions = mcp.build_instructions_prompt()
    if instructions:
        print(f"MCP server instructions:\n{instructions[:200]}...")

    # --- Create engine with built-in + MCP tools ---
    engine = Engine(config=config, tools=get_default_tools())

    # Discover and register MCP tools
    mcp_tools = await mcp.discover_tools_async(engine.registry)
    print(f"Discovered {len(mcp_tools)} MCP tools:")
    for t in mcp_tools:
        print(f"  - {t.name}: {t.description[:60]}")

    # Register MCP resource tools (for browsing server resources)
    engine.registry.register(ListMcpResourcesTool(mcp_manager=mcp))
    engine.registry.register(ReadMcpResourceTool(mcp_manager=mcp))

    # --- Use the agent (MCP tools + resources are now available) ---
    async for event in engine.run("List files in /tmp using the filesystem tool"):
        if event["type"] == "tool_call":
            print(f"  ⚡ {event['name']}")
        elif event["type"] == "done":
            print(f"\n{event['content']}")

    # --- Cleanup ---
    await mcp.close()


if __name__ == "__main__":
    asyncio.run(main())
