"""
CLI Entry — 命令行交互入口
Maps to: src/main.tsx → REPL mode

Provider-agnostic REPL: auto-detects provider from env vars.
Provider 无关的 REPL：从环境变量自动检测 Provider。
"""
import asyncio
import os
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from claw_agent.config import Config
from claw_agent.core.engine import Engine
from claw_agent.core.mcp_client import MCPManager, MCPServerConfig
from claw_agent.core.provider import PROVIDER_PRESETS
from claw_agent.tools import get_default_tools
from claw_agent.memory.memory import Memory


console = Console()


def _detect_config() -> Config:
    """Auto-detect provider from environment variables / 从环境变量自动检测 Provider
    User only needs to set ONE env var (e.g. MINIMAX_API_KEY) — we figure out the rest.
    """
    # Check each provider's env var in priority order
    priority = ["minimax", "openai", "anthropic", "gemini", "kimi", "deepseek", "qwen"]
    for name in priority:
        preset = PROVIDER_PRESETS[name]
        env_key = preset["env_key"]
        key = os.environ.get(env_key, "")
        if key:
            return Config(provider=name, api_key=key)

    # No key found — show help
    console.print("[red]No API key found.[/red]\n")
    table = Table(title="Supported Providers / 支持的模型供应商", show_lines=True)
    table.add_column("Provider", style="cyan")
    table.add_column("Env Variable", style="yellow")
    table.add_column("Default Model", style="green")
    for name in priority:
        p = PROVIDER_PRESETS[name]
        table.add_row(name, p["env_key"], p["model"])
    console.print(table)
    console.print("\n[dim]Set one of the above env vars and try again.[/dim]")
    sys.exit(1)


async def repl():
    """Interactive REPL / 交互式会话"""
    config = _detect_config()

    # Initialize tools
    tools = get_default_tools()

    # Initialize memory if enabled
    if config.feature("MEMORY"):
        memory = Memory(config.effective_memory_dir)
        memory_prompt = memory.build_prompt()
        config.append_system_prompt = memory_prompt

    # Initialize MCP if configured
    mcp_manager = None
    if config.feature("MCP") and config.mcp_servers:
        mcp_manager = MCPManager()
        servers = [MCPServerConfig(**s) for s in config.mcp_servers]
        await mcp_manager.connect_all(servers)

    # Create engine
    engine = Engine(config=config, tools=tools)

    # Discover MCP tools
    if mcp_manager:
        await mcp_manager.discover_tools_async(engine.registry)

    tool_count = len(engine.registry.all())

    # Banner
    console.print(Panel.fit(
        f"[bold cyan]Claw Agent[/bold cyan] — Python Agent Framework\n"
        f"[dim]Provider: [bold]{config.provider}[/bold] | "
        f"Model: [bold]{config.effective_model}[/bold] | "
        f"Tools: {tool_count} | "
        f"Memory: {'on' if config.feature('MEMORY') else 'off'}[/dim]",
        border_style="cyan",
    ))
    console.print("[dim]Type your message. /quit to exit, /tools to list, /reset to clear.[/dim]\n")

    try:
        while True:
            try:
                user_input = console.input("[bold green]> [/bold green]")
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input.strip():
                continue

            # Special commands
            if user_input.strip() == "/quit":
                break
            if user_input.strip() == "/reset":
                engine.reset()
                console.print("[dim]Conversation reset.[/dim]")
                continue
            if user_input.strip() == "/tools":
                for t in engine.registry.all():
                    console.print(f"  [cyan]{t.name}[/cyan] — {t.description[:60]}")
                continue
            if user_input.strip() == "/provider":
                console.print(f"  Provider: [cyan]{config.provider}[/cyan]")
                console.print(f"  Model: [cyan]{config.effective_model}[/cyan]")
                continue

            # Run agent loop
            async for event in engine.run(user_input):
                if event["type"] == "thinking":
                    console.print(f"  [dim italic]💭 {_truncate(event['content'], 100)}[/dim italic]")
                elif event["type"] == "tool_call":
                    console.print(
                        f"  [yellow]⚡ {event['name']}[/yellow]"
                        f"[dim]({_truncate(str(event['arguments']), 80)})[/dim]"
                    )
                elif event["type"] == "tool_result":
                    output = _truncate(event["content"], 200)
                    console.print(f"  [dim]→ {output}[/dim]")
                elif event["type"] == "done":
                    console.print()
                    console.print(Markdown(event["content"]))
                    console.print()
                elif event["type"] == "error":
                    console.print(f"[red]Error: {event['content']}[/red]")

    finally:
        if mcp_manager:
            await mcp_manager.close()
        console.print("\n[dim]Bye![/dim]")


def _truncate(s: str, max_len: int) -> str:
    return s[:max_len] + "..." if len(s) > max_len else s


def main():
    asyncio.run(repl())


if __name__ == "__main__":
    main()
