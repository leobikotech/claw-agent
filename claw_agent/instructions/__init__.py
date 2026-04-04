"""
Instructions — 指令 + 系统提示词
Maps to: src/utils/claudemd.ts, src/prompt.ts

Discovers CLAW.md instruction files and builds the system prompt.
发现 CLAW.md 指令文件并构建系统提示词。
"""
from claw_agent.instructions.clawmd import (
    InstructionFile,
    discover_instruction_files,
    format_instructions_prompt,
    get_user_claw_dir,
    get_large_files,
)
from claw_agent.instructions.prompts import PromptBuilder

__all__ = [
    "InstructionFile",
    "discover_instruction_files",
    "format_instructions_prompt",
    "get_user_claw_dir",
    "get_large_files",
    "PromptBuilder",
]
