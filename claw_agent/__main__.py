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
from claw_agent.core.hooks import HookContext, HookEvent, HookManager
from claw_agent.integrations.mcp_client import MCPManager, MCPServerConfig
from claw_agent.tools.mcp_resources import ListMcpResourcesTool, ReadMcpResourceTool
from claw_agent.providers import PROVIDER_PRESETS
from claw_agent.tools import get_default_tools
from claw_agent.memory.memory import Memory
from claw_agent.memory.dream import (
    DreamEngine, DreamConfig,
    create_dream_hook, create_session_record_hook,
)


console = Console()


def _parse_args():
    """Parse CLI arguments / 解析命令行参数"""
    import argparse
    parser = argparse.ArgumentParser(
        prog="claw",
        description="Claw Agent — Python Agent Framework",
    )
    parser.add_argument(
        "--language", type=str, default=None,
        help="Preferred language for agent responses (e.g. japanese, chinese, spanish). "
             "Also reads from CLAW_LANGUAGE env var.",
    )
    parser.add_argument(
        "--provider", type=str, default=None,
        help="LLM provider override (openai, anthropic, gemini, minimax, etc.)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model override (e.g. gpt-4o, claude-sonnet-4-20250514)",
    )
    return parser.parse_args()


def _detect_config(args=None) -> Config:
    """Auto-detect provider from environment variables / 从环境变量自动检测 Provider
    User only needs to set ONE env var (e.g. MINIMAX_API_KEY) — we figure out the rest.
    """
    # Check each provider's env var in priority order
    priority = ["minimax", "openai", "anthropic", "gemini", "kimi", "deepseek", "qwen"]

    # If provider explicitly specified via CLI
    if args and args.provider:
        preset = PROVIDER_PRESETS.get(args.provider.lower())
        if preset:
            key = os.environ.get(preset["env_key"], "")
            return Config(provider=args.provider.lower(), api_key=key)

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
    args = _parse_args()
    config = _detect_config(args)

    # Apply CLI overrides
    if args.language:
        config.language = args.language
    if args.model:
        config.model = args.model

    # Initialize tools
    tools = get_default_tools()

    # Initialize memory if enabled
    memory = None
    dream_engine = None
    if config.feature("MEMORY"):
        memory = Memory(config.effective_memory_dir)
        memory_prompt = memory.build_prompt()
        config.append_system_prompt = memory_prompt

    # Initialize HookManager
    hook_manager = HookManager()

    # Initialize Dream engine + register hooks if enabled
    # Maps to: initAutoDream() in autoDream.ts + stopHooks.ts:155
    if config.feature("DREAM") and memory:
        dream_engine = DreamEngine(
            memory=memory,
            app_config=config,
        )
        # STOP hook: fire-and-forget dream trigger (maps to `void executeAutoDream()`)
        hook_manager.register(
            HookEvent.STOP,
            create_dream_hook(dream_engine),
            name="auto_dream",
            fire_and_forget=True,
        )
        # SESSION_END hook: increment session counter for Gate 2
        hook_manager.register(
            HookEvent.SESSION_END,
            create_session_record_hook(dream_engine),
            name="session_record",
        )

    # Initialize MCP if configured
    mcp_manager = None
    if config.feature("MCP") and config.mcp_servers:
        mcp_manager = MCPManager()
        servers = [MCPServerConfig(**s) for s in config.mcp_servers]
        await mcp_manager.connect_all(servers)

        # --- P0 fix: inject MCP server instructions into system prompt ---
        # Maps to: server instructions injection in client.ts
        mcp_instructions = mcp_manager.build_instructions_prompt()
        if mcp_instructions:
            existing = config.append_system_prompt or ""
            config.append_system_prompt = existing + "\n\n" + mcp_instructions if existing else mcp_instructions

    # Create engine with hook manager
    engine = Engine(config=config, tools=tools, memory=memory, hook_manager=hook_manager)

    # Discover MCP tools + register MCP resource tools
    if mcp_manager:
        await mcp_manager.discover_tools_async(engine.registry)
        # --- P1 fix: register MCP resource tools ---
        # Maps to: ListMcpResourcesTool + ReadMcpResourceTool in src/tools/
        engine.registry.register(ListMcpResourcesTool(mcp_manager=mcp_manager))
        engine.registry.register(ReadMcpResourceTool(mcp_manager=mcp_manager))

    tool_count = len(engine.registry.all())

    # Banner
    lang_display = f"Language: [bold]{config.language}[/bold] | " if config.language else ""
    console.print(Panel.fit(
        f"[bold cyan]Claw Agent[/bold cyan] — Python Agent Framework\n"
        f"[dim]Provider: [bold]{config.provider}[/bold] | "
        f"Model: [bold]{config.effective_model}[/bold] | "
        f"Tools: {tool_count} | "
        f"{lang_display}"
        f"Memory: {'on' if config.feature('MEMORY') else 'off'} | "
        f"Dream: {'on' if config.feature('DREAM') and dream_engine else 'off'} | "
        f"Hooks: {sum(len(v) for v in hook_manager.list_hooks().values())}[/dim]",
        border_style="cyan",
    ))
    console.print("[dim]Type your message. /quit to exit, /tools to list, /reset to clear, /dream to consolidate, /hooks to inspect.[/dim]\n")

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

            # /dream — manual dream consolidation
            # Maps to: /dream command in commands/dream/index.ts
            if user_input.strip() == "/dream":
                if dream_engine:
                    console.print("  [dim]💤 Running dream consolidation...[/dim]")
                    result = await dream_engine.run_force()
                    console.print(f"  [magenta]{result}[/magenta]")
                else:
                    console.print("[yellow]Dream not enabled. Set DREAM feature to True.[/yellow]")
                continue

            # /hooks — inspect registered hooks
            # Maps to: /hooks command in commands/hooks/index.ts
            if user_input.strip() == "/hooks":
                hooks_info = hook_manager.list_hooks()
                if not hooks_info:
                    console.print("  [dim]No hooks registered.[/dim]")
                else:
                    for event_name, hooks_list in hooks_info.items():
                        console.print(f"  [cyan]{event_name}[/cyan]:")
                        for h in hooks_list:
                            mode = "🔥 fire-and-forget" if h['fire_and_forget'] else "🔒 sync"
                            console.print(
                                f"    [dim]• {h['name']} (priority={h['priority']}, {mode})[/dim]"
                            )
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
                elif event["type"] == "partial":
                    console.print()
                    console.print(Markdown(event["content"]))
                elif event["type"] == "waiting":
                    console.print(f"  [dim]⏳ {event['content']}[/dim]")
                elif event["type"] == "worker_notification":
                    console.print(f"  [magenta]📨 Worker notification received[/magenta]")
                elif event["type"] == "error":
                    console.print(f"[red]Error: {event['content']}[/red]")

    finally:
        # --- SESSION_END hooks (maps to executeSessionEndHooks in hooks.ts) ---
        session_end_ctx = HookContext(
            messages=list(engine.messages),
            engine=engine,
        )
        await hook_manager.execute(HookEvent.SESSION_END, session_end_ctx)
        await hook_manager.shutdown()  # Wait for fire-and-forget tasks

        if mcp_manager:
            await mcp_manager.close()
        console.print("\n[dim]Bye![/dim]")


def _truncate(s: str, max_len: int) -> str:
    return s[:max_len] + "..." if len(s) > max_len else s


def main():
    asyncio.run(repl())


if __name__ == "__main__":
    main()
