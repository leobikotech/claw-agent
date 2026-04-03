"""
Memory & Dream Example — 持久化记忆与睡眠做梦示例
Shows how to use file-based memory injection and memory consolidation (Dream).
展示如何使用基于文件的记忆注入和记忆巩固（做梦机制）。
"""
import asyncio
import os
from claw_agent import Engine, Config
from claw_agent.tools import get_default_tools
from claw_agent.memory import Memory, DreamEngine, DreamConfig


async def main():
    config = Config()

    # --- 1. Initialize Memory ---
    # Create persistent memory in current local directory
    mem_dir = os.path.join(os.getcwd(), ".claw_memory")
    memory = Memory(mem_dir)

    print(f"Memory directory created at: {mem_dir}")

    # Seed an artificial memory manually for the example
    # In reality, the Agent/LLM uses FileWriteTool to save stuff and edits MEMORY.md
    memory.save(
        title="User Preferences",
        content="The user strictly prefers Python 3.12+ features. They hate list comprehensions.",
        filename="user_prefs.md"
    )

    # --- 2. Initialize Engine with Memory ---
    # Default tools include File tools, Bash, Glob, Grep, etc.
    engine = Engine(
        config=config,
        tools=get_default_tools(),
        memory=memory  # Pass memory instance to Engine
    )

    # When we run this prompt, the Engine will automatically call
    # find_relevant_memories() and inject "user_prefs.md" as a <system-reminder>
    print("\n--- Example 1: Relevant Memory Injection ---")
    async for event in engine.run("Write a loop to generate a list of squares from 1 to 5.", max_turns=1):
        if event["type"] == "thinking":
            pass
        elif event["type"] == "tool_call":
            print(f"  ⚡ {event['name']} called")
        elif event["type"] == "compact":
            print("  [Auto-Compact Triggered]")
        elif event["type"] == "done":
            print(f"\nFinal Result:\n{event['content']}")

    # Check the actual LLM prompt to see our injected attachment
    print("\n[Debug] First Turn User Prompt Content:")
    print(engine.messages[-2].content[:200] + "...\n")

    # --- 3. Memory Consolidation (Dream) ---
    print("\n--- Example 2: Memory Consolidation (AutoDream) ---")
    print("Now taking the agent offline to compact today's context into long-term memories...")

    dream_config = DreamConfig(
        memory_dir=mem_dir,
        topics=["python features", "user behavior", "coding style"],
        max_duration_seconds=10
    )

    # Build the Dream engine using the same provider
    dream_engine = DreamEngine(config, dream_config, provider=engine._provider)
    
    # We pass the conversation history we just had into the dream generator
    await dream_engine.dream(context_files=[], conversation_history=engine.messages)
    print("\nDream sequence complete! It might have consolidated our learning into a new note.")

    # Show memory index contents
    print("\nCurrent Persistent Memory Index:")
    print(memory.load_index())


if __name__ == "__main__":
    asyncio.run(main())
