"""
MCP Example — Model Context Protocol 示例
Connect to an MCP server and use its tools in the agent.
连接 MCP 服务器并在智能体中使用其工具。
"""
import asyncio
from claw_agent import Engine, Config
from claw_agent.tools import get_default_tools
from claw_agent.core.mcp_client import MCPManager, MCPServerConfig


async def main():
    config = Config()

    # --- Define MCP servers to connect ---
    # Example: a filesystem MCP server
    servers = [
        MCPServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        ),
        # Add more MCP servers here:
        # MCPServerConfig(name="github", command="...", args=[...]),
    ]

    # --- Connect to MCP servers ---
    mcp = MCPManager()
    await mcp.connect_all(servers)

    # --- Create engine with built-in + MCP tools ---
    engine = Engine(config=config, tools=get_default_tools())

    # Discover and register MCP tools
    mcp_tools = await mcp.discover_tools_async(engine.registry)
    print(f"Discovered {len(mcp_tools)} MCP tools:")
    for t in mcp_tools:
        print(f"  - {t.name}: {t.description[:60]}")

    # --- Use the agent (MCP tools are now available) ---
    async for event in engine.run("List files in /tmp using the filesystem tool"):
        if event["type"] == "tool_call":
            print(f"  ⚡ {event['name']}")
        elif event["type"] == "done":
            print(f"\n{event['content']}")

    # --- Cleanup ---
    await mcp.close()


if __name__ == "__main__":
    asyncio.run(main())
