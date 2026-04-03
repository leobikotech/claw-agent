"""
Minimal Agent Example — 最小智能体示例
10 lines to start a working agent — provider auto-detected from env.
10 行代码启动智能体——Provider 从环境变量自动检测。

Just set ONE env var:
  export MINIMAX_API_KEY=your_key   # MiniMax M2.7
  export OPENAI_API_KEY=your_key    # OpenAI GPT-4o
  export ANTHROPIC_API_KEY=your_key # Claude
"""
import asyncio
from claw_agent import Engine, Config
from claw_agent.tools import get_default_tools


async def main():
    # Auto-detects provider from env. Or specify explicitly:
    # config = Config(provider="minimax", api_key="your_key")
    # config = Config(provider="openai",  api_key="sk-...")
    # config = Config(provider="anthropic", api_key="sk-ant-...")
    config = Config(provider="minimax")  # reads MINIMAX_API_KEY from env
    engine = Engine(config=config, tools=get_default_tools())

    async for event in engine.run("List all Python files in the current directory"):
        if event["type"] == "tool_call":
            print(f"  ⚡ {event['name']}({event['arguments']})")
        elif event["type"] == "done":
            print(f"\n{event['content']}")


if __name__ == "__main__":
    asyncio.run(main())
