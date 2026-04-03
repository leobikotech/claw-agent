"""
Multi-Provider Example — 多模型切换示例
Same code, different providers. Just swap the config.
相同代码，不同 Provider。只需切换配置。
"""
import asyncio
from claw_agent import Engine, Config
from claw_agent.tools import get_default_tools


async def run_with_provider(provider: str, api_key: str):
    """Run the same prompt with a specific provider"""
    config = Config(provider=provider, api_key=api_key)
    engine = Engine(config=config, tools=get_default_tools())

    print(f"\n{'='*50}")
    print(f"Provider: {provider} | Model: {config.effective_model}")
    print(f"{'='*50}")

    async for event in engine.run("What is 2+2? Use bash to verify with python3 -c 'print(2+2)'"):
        if event["type"] == "tool_call":
            print(f"  ⚡ {event['name']}")
        elif event["type"] == "done":
            print(f"  Result: {event['content'][:200]}")


async def main():
    import os

    # Run with whichever providers you have keys for
    providers = {
        "minimax":   os.environ.get("MINIMAX_API_KEY"),
        "openai":    os.environ.get("OPENAI_API_KEY"),
        "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
        "gemini":    os.environ.get("GEMINI_API_KEY"),
        "kimi":      os.environ.get("KIMI_API_KEY"),
        "deepseek":  os.environ.get("DEEPSEEK_API_KEY"),
    }

    available = {k: v for k, v in providers.items() if v}

    if not available:
        print("No API keys found. Set at least one:")
        print("  export MINIMAX_API_KEY=...")
        print("  export OPENAI_API_KEY=...")
        return

    print(f"Found {len(available)} provider(s): {', '.join(available.keys())}")

    for provider, key in available.items():
        await run_with_provider(provider, key)


if __name__ == "__main__":
    asyncio.run(main())
