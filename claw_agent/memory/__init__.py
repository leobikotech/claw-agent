"""
Memory — 记忆系统
File-based persistent memory with dream consolidation and auto-compact.
"""
from claw_agent.memory.memory import (
    Memory, MemoryHeader, MemoryEntry,
    scan_memory_files, format_memory_manifest,
    find_relevant_memories,
)
from claw_agent.memory.dream import DreamEngine, DreamConfig
from claw_agent.memory.compact import (
    auto_compact_if_needed, should_auto_compact,
    CompactTrackingState, get_compact_prompt,
)
