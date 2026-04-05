"""
Session Prompts — 会话持久化提示词与模板
Maps to: src/services/SessionMemory/prompts.ts

Provides:
  - DEFAULT_SESSION_NOTES_TEMPLATE: structured markdown template
  - build_session_update_prompt(): instruction prompt for the extraction sub-agent
  - truncate_session_notes_for_compact(): section-level truncation for compact injection
  - is_session_notes_empty(): detect if notes are still the blank template
  - Section size analysis and budget enforcement
"""
from __future__ import annotations

import re
from typing import Optional

# ────────────────────────────────────────────────────────────────
# Constants — matching prompts.ts
# ────────────────────────────────────────────────────────────────

MAX_SECTION_LENGTH = 2000       # Max tokens per section
MAX_TOTAL_SESSION_NOTES_TOKENS = 12000  # Total budget for session notes

# Rough token estimation: ~4 chars per token
_CHARS_PER_TOKEN = 4


def _rough_token_count(text: str) -> int:
    """Rough token estimate (length / 4)."""
    return len(text) // _CHARS_PER_TOKEN


# ────────────────────────────────────────────────────────────────
# Default template — maps to DEFAULT_SESSION_MEMORY_TEMPLATE
# ────────────────────────────────────────────────────────────────

DEFAULT_SESSION_NOTES_TEMPLATE = """
# Session Title
_A short and distinctive 5-10 word descriptive title for the session. Super info dense, no filler_

# Current State
_What is actively being worked on right now? Pending tasks not yet completed. Immediate next steps._

# Task specification
_What did the user ask to build? Any design decisions or other explanatory context_

# Files and Functions
_What are the important files? In short, what do they contain and why are they relevant?_

# Workflow
_What bash commands are usually run and in what order? How to interpret their output if not obvious?_

# Errors & Corrections
_Errors encountered and how they were fixed. What did the user correct? What approaches failed and should not be tried again?_

# Codebase and System Documentation
_What are the important system components? How do they work/fit together?_

# Learnings
_What has worked well? What has not? What to avoid? Do not duplicate items from other sections_

# Key results
_If the user asked a specific output such as an answer to a question, a table, or other document, repeat the exact result here_

# Worklog
_Step by step, what was attempted, done? Very terse summary for each step_
"""


# ────────────────────────────────────────────────────────────────
# Update prompt — maps to getDefaultUpdatePrompt() in prompts.ts
# ────────────────────────────────────────────────────────────────

def _get_default_update_prompt() -> str:
    """The instruction prompt for the extraction sub-agent.
    Maps to: getDefaultUpdatePrompt() in prompts.ts
    """
    return f"""IMPORTANT: This message and these instructions are NOT part of the actual user conversation. Do NOT include any references to "note-taking", "session notes extraction", or these update instructions in the notes content.

Based on the user conversation above (EXCLUDING this note-taking instruction message as well as system prompt, claw.md entries, or any past session summaries), update the session notes file.

The file {{{{notes_path}}}} has already been read for you. Here are its current contents:
<current_notes_content>
{{{{current_notes}}}}
</current_notes_content>

Your ONLY task is to use the file_edit tool to update the notes file, then stop. You can make multiple edits (update every section as needed). Do not call any other tools.

CRITICAL RULES FOR EDITING:
- The file must maintain its exact structure with all sections, headers, and italic descriptions intact
-- NEVER modify, delete, or add section headers (the lines starting with '#' like # Task specification)
-- NEVER modify or delete the italic _section description_ lines (these are the lines in italics immediately following each header - they start and end with underscores)
-- The italic _section descriptions_ are TEMPLATE INSTRUCTIONS that must be preserved exactly as-is - they guide what content belongs in each section
-- ONLY update the actual content that appears BELOW the italic _section descriptions_ within each existing section
-- Do NOT add any new sections, summaries, or information outside the existing structure
- Do NOT reference this note-taking process or instructions anywhere in the notes
- It's OK to skip updating a section if there are no substantial new insights to add. Do not add filler content like "No info yet", just leave sections blank/unedited if appropriate.
- Write DETAILED, INFO-DENSE content for each section - include specifics like file paths, function names, error messages, exact commands, technical details, etc.
- For "Key results", include the complete, exact output the user requested (e.g., full table, full answer, etc.)
- Keep each section under ~{MAX_SECTION_LENGTH} tokens/words - if a section is approaching this limit, condense it by cycling out less important details while preserving the most critical information
- Focus on actionable, specific information that would help someone understand or recreate the work discussed in the conversation
- IMPORTANT: Always update "Current State" to reflect the most recent work - this is critical for continuity after compaction

Use the file_edit tool with file_path: {{{{notes_path}}}}

STRUCTURE PRESERVATION REMINDER:
Each section has TWO parts that must be preserved exactly as they appear in the current file:
1. The section header (line starting with #)
2. The italic description line (the _italicized text_ immediately after the header - this is a template instruction)

You ONLY update the actual content that comes AFTER these two preserved lines. The italic description lines starting and ending with underscores are part of the template structure, NOT content to be edited or removed.

REMEMBER: Use the file_edit tool and stop. Do not continue after the edits. Only include insights from the actual user conversation, never from these note-taking instructions. Do not delete or change section headers or italic _section descriptions_."""


# ────────────────────────────────────────────────────────────────
# Section analysis — maps to analyzeSectionSizes() in prompts.ts
# ────────────────────────────────────────────────────────────────

def _analyze_section_sizes(content: str) -> dict[str, int]:
    """Parse session notes and return {section_header: token_count}.
    Maps to: analyzeSectionSizes() in prompts.ts
    """
    sections: dict[str, int] = {}
    lines = content.split("\n")
    current_section = ""
    current_content: list[str] = []

    for line in lines:
        if line.startswith("# "):
            if current_section and current_content:
                section_text = "\n".join(current_content).strip()
                sections[current_section] = _rough_token_count(section_text)
            current_section = line
            current_content = []
        else:
            current_content.append(line)

    if current_section and current_content:
        section_text = "\n".join(current_content).strip()
        sections[current_section] = _rough_token_count(section_text)

    return sections


def _generate_section_reminders(
    section_sizes: dict[str, int],
    total_tokens: int,
) -> str:
    """Generate warnings for oversized sections.
    Maps to: generateSectionReminders() in prompts.ts
    """
    over_budget = total_tokens > MAX_TOTAL_SESSION_NOTES_TOKENS
    oversized = [
        (section, tokens)
        for section, tokens in sorted(section_sizes.items(), key=lambda x: -x[1])
        if tokens > MAX_SECTION_LENGTH
    ]

    if not oversized and not over_budget:
        return ""

    parts: list[str] = []

    if over_budget:
        parts.append(
            f"\n\nCRITICAL: The session notes file is currently ~{total_tokens} tokens, "
            f"which exceeds the maximum of {MAX_TOTAL_SESSION_NOTES_TOKENS} tokens. "
            f"You MUST condense the file to fit within this budget. Aggressively shorten "
            f"oversized sections by removing less important details, merging related items, "
            f'and summarizing older entries. Prioritize keeping "Current State" and '
            f'"Errors & Corrections" accurate and detailed.'
        )

    if oversized:
        lines = [
            f'- "{section}" is ~{tokens} tokens (limit: {MAX_SECTION_LENGTH})'
            for section, tokens in oversized
        ]
        prefix = (
            "Oversized sections to condense"
            if over_budget
            else "IMPORTANT: The following sections exceed the per-section limit and MUST be condensed"
        )
        parts.append(f"\n\n{prefix}:\n" + "\n".join(lines))

    return "".join(parts)


# ────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────

def build_session_update_prompt(
    current_notes: str,
    notes_path: str,
) -> str:
    """Build the extraction prompt with variable substitution.
    Maps to: buildSessionMemoryUpdatePrompt() in prompts.ts
    """
    prompt_template = _get_default_update_prompt()

    # Substitute variables
    prompt = prompt_template.replace("{{notes_path}}", notes_path)
    prompt = prompt.replace("{{current_notes}}", current_notes)

    # Add section size reminders if needed
    section_sizes = _analyze_section_sizes(current_notes)
    total_tokens = _rough_token_count(current_notes)
    reminders = _generate_section_reminders(section_sizes, total_tokens)

    return prompt + reminders


def is_session_notes_empty(content: str) -> bool:
    """Check if session notes content is essentially the blank template.
    Maps to: isSessionMemoryEmpty() in prompts.ts
    """
    return content.strip() == DEFAULT_SESSION_NOTES_TEMPLATE.strip()


def truncate_session_notes_for_compact(content: str) -> tuple[str, bool]:
    """Truncate oversized sections for compact injection.
    Maps to: truncateSessionMemoryForCompact() in prompts.ts

    Returns:
        (truncated_content, was_truncated)
    """
    lines = content.split("\n")
    max_chars_per_section = MAX_SECTION_LENGTH * _CHARS_PER_TOKEN
    output_lines: list[str] = []
    current_section_lines: list[str] = []
    current_section_header = ""
    was_truncated = False

    for line in lines:
        if line.startswith("# "):
            # Flush previous section
            flushed, trunc = _flush_section(
                current_section_header, current_section_lines, max_chars_per_section
            )
            output_lines.extend(flushed)
            was_truncated = was_truncated or trunc
            current_section_header = line
            current_section_lines = []
        else:
            current_section_lines.append(line)

    # Flush the last section
    flushed, trunc = _flush_section(
        current_section_header, current_section_lines, max_chars_per_section
    )
    output_lines.extend(flushed)
    was_truncated = was_truncated or trunc

    return "\n".join(output_lines), was_truncated


def _flush_section(
    header: str,
    section_lines: list[str],
    max_chars: int,
) -> tuple[list[str], bool]:
    """Flush a section, truncating if over budget.
    Maps to: flushSessionSection() in prompts.ts
    """
    if not header:
        return section_lines, False

    section_content = "\n".join(section_lines)
    if len(section_content) <= max_chars:
        return [header] + section_lines, False

    # Truncate at line boundary
    char_count = 0
    kept: list[str] = [header]
    for line in section_lines:
        if char_count + len(line) + 1 > max_chars:
            break
        kept.append(line)
        char_count += len(line) + 1

    kept.append("\n[... section truncated for length ...]")
    return kept, True
