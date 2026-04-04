"""
Compact — 上下文压缩服务
Maps to: src/services/compact/ (autoCompact.ts, compact.ts, prompt.ts)

Automatically compresses conversation history when approaching context window
limits, using the LLM itself to generate structured summaries.

Architecture:
  1. Token estimation (tokens.py) → check threshold
  2. If above threshold → build compact prompt → LLM generates <analysis>+<summary>
  3. Replace old messages with compact summary user message
  4. Continue conversation with compressed history
"""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from claw_agent.providers.tokens import (
    estimate_tokens_messages,
    get_context_window,
)

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────
# Constants — matching autoCompact.ts
# ────────────────────────────────────────────────────────────────

# Reserve this many tokens for the compact summary output
MAX_OUTPUT_TOKENS_FOR_SUMMARY = 20_000

# Buffer tokens before triggering autocompact
AUTOCOMPACT_BUFFER_TOKENS = 13_000

# Warning thresholds
WARNING_THRESHOLD_BUFFER = 20_000
ERROR_THRESHOLD_BUFFER = 20_000

# Manual compact needs less buffer
MANUAL_COMPACT_BUFFER = 3_000

# Circuit breaker: stop after N consecutive failures
MAX_CONSECUTIVE_FAILURES = 3


# ────────────────────────────────────────────────────────────────
# Tracking state — maps to AutoCompactTrackingState
# ────────────────────────────────────────────────────────────────

@dataclass
class CompactTrackingState:
    """Tracks compact state across turns."""
    compacted: bool = False
    turn_counter: int = 0
    consecutive_failures: int = 0


# ────────────────────────────────────────────────────────────────
# Compact prompts — 1:1 translation of prompt.ts (375 lines)
# ────────────────────────────────────────────────────────────────

NO_TOOLS_PREAMBLE = """CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

- Do NOT use Read, Bash, Grep, Glob, Edit, Write, or ANY other tool.
- You already have all the context you need in the conversation above.
- Tool calls will be REJECTED and will waste your only turn — you will fail the task.
- Your entire response must be plain text: an <analysis> block followed by a <summary> block.

"""

DETAILED_ANALYSIS_INSTRUCTION = """Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly."""

BASE_COMPACT_PROMPT = f"""Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

{DETAILED_ANALYSIS_INSTRUCTION}

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
9. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most recent explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests or really old requests that were already completed without confirming with the user first.
                       If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.

Here's an example of how your output should be structured:

<example>
<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]
   - [...]

3. Files and Code Sections:
   - [File Name 1]
      - [Summary of why this file is important]
      - [Summary of the changes made to this file, if any]
      - [Important Code Snippet]
   - [File Name 2]
      - [Important Code Snippet]
   - [...]

4. Errors and fixes:
    - [Detailed description of error 1]:
      - [How you fixed the error]
      - [User feedback on the error if any]
    - [...]

5. Problem Solving:
   [Description of solved problems and ongoing troubleshooting]

6. All user messages:
    - [Detailed non tool use user message]
    - [...]

7. Pending Tasks:
   - [Task 1]
   - [Task 2]
   - [...]

8. Current Work:
   [Precise description of current work]

9. Optional Next Step:
   [Optional Next step to take]

</summary>
</example>

Please provide your summary based on the conversation so far, following this structure and ensuring precision and thoroughness in your response.

There may be additional summarization instructions provided in the included context. If so, remember to follow these instructions when creating the above summary. Examples of instructions include:
<example>
## Compact Instructions
When summarizing the conversation focus on typescript code changes and also remember the mistakes you made and how you fixed them.
</example>

<example>
# Summary instructions
When you are using compact - please focus on test output and code changes. Include file reads verbatim.
</example>
"""

NO_TOOLS_TRAILER = (
    "\n\nREMINDER: Do NOT call any tools. Respond with plain text only — "
    "an <analysis> block followed by a <summary> block. "
    "Tool calls will be rejected and you will fail the task."
)


def get_compact_prompt(custom_instructions: Optional[str] = None) -> str:
    """Build the full compact prompt for the LLM.
    Maps to: getCompactPrompt() in prompt.ts
    """
    prompt = NO_TOOLS_PREAMBLE + BASE_COMPACT_PROMPT
    if custom_instructions and custom_instructions.strip():
        prompt += f"\n\nAdditional Instructions:\n{custom_instructions}"
    prompt += NO_TOOLS_TRAILER
    return prompt


def format_compact_summary(raw_summary: str) -> str:
    """Strip <analysis> scratchpad and format <summary> tags.
    Maps to: formatCompactSummary() in prompt.ts
    """
    result = raw_summary

    # Strip analysis section — drafting scratchpad, not needed in final output
    result = re.sub(r"<analysis>[\s\S]*?</analysis>", "", result)

    # Extract and format summary section
    match = re.search(r"<summary>([\s\S]*?)</summary>", result)
    if match:
        content = match.group(1).strip()
        result = re.sub(r"<summary>[\s\S]*?</summary>", f"Summary:\n{content}", result)

    # Clean up whitespace
    result = re.sub(r"\n\n+", "\n\n", result)
    return result.strip()


def get_compact_user_summary_message(
    summary: str,
    suppress_followup: bool = False,
) -> str:
    """Build the user message that replaces compacted history.
    Maps to: getCompactUserSummaryMessage() in prompt.ts
    """
    formatted = format_compact_summary(summary)

    base = (
        "This session is being continued from a previous conversation that ran "
        "out of context. The summary below covers the earlier portion of the "
        f"conversation.\n\n{formatted}"
    )

    if suppress_followup:
        base += (
            "\nContinue the conversation from where it left off without asking "
            "the user any further questions. Resume directly — do not acknowledge "
            "the summary, do not recap what was happening, do not preface with "
            '"I\'ll continue" or similar. Pick up the last task as if the break '
            "never happened."
        )

    return base


# ────────────────────────────────────────────────────────────────
# Auto-compact logic — maps to autoCompact.ts
# ────────────────────────────────────────────────────────────────

def get_effective_context_window(model: str) -> int:
    """Context window minus reserved output tokens.
    Maps to: getEffectiveContextWindowSize() in autoCompact.ts
    """
    context_window = get_context_window(model)
    reserved = min(MAX_OUTPUT_TOKENS_FOR_SUMMARY, 8_192)  # Conservative for our providers
    return context_window - reserved


def get_auto_compact_threshold(model: str) -> int:
    """Threshold at which autocompact triggers.
    Maps to: getAutoCompactThreshold() in autoCompact.ts
    """
    effective = get_effective_context_window(model)
    return effective - AUTOCOMPACT_BUFFER_TOKENS


def should_auto_compact(
    messages: list[dict[str, Any]],
    model: str,
) -> bool:
    """Check if conversation should be auto-compacted.
    Maps to: shouldAutoCompact() in autoCompact.ts
    """
    token_count = estimate_tokens_messages(messages)
    threshold = get_auto_compact_threshold(model)

    logger.debug(
        f"autocompact check: tokens≈{token_count} threshold={threshold} "
        f"model={model}"
    )

    return token_count >= threshold


def calculate_token_warning_state(
    messages: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    """Calculate context usage warnings.
    Maps to: calculateTokenWarningState() in autoCompact.ts
    """
    token_count = estimate_tokens_messages(messages)
    threshold = get_auto_compact_threshold(model)
    effective = get_effective_context_window(model)

    percent_left = max(0, round(((threshold - token_count) / threshold) * 100))
    warning_threshold = threshold - WARNING_THRESHOLD_BUFFER
    error_threshold = threshold - ERROR_THRESHOLD_BUFFER
    blocking_limit = effective - MANUAL_COMPACT_BUFFER

    return {
        "token_count": token_count,
        "percent_left": percent_left,
        "is_above_warning": token_count >= warning_threshold,
        "is_above_error": token_count >= error_threshold,
        "is_above_auto_compact": token_count >= threshold,
        "is_at_blocking_limit": token_count >= blocking_limit,
    }


async def auto_compact_if_needed(
    messages: list[dict[str, Any]],
    model: str,
    provider: Any,  # LLMProvider
    tracking: Optional[CompactTrackingState] = None,
    custom_instructions: Optional[str] = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Run autocompact if needed. Returns (new_messages, was_compacted).
    Maps to: autoCompactIfNeeded() in autoCompact.ts

    This is the main entry point called by Engine after each turn.
    """
    if tracking and tracking.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
        logger.warning("autocompact: circuit breaker tripped, skipping")
        return messages, False

    if not should_auto_compact(messages, model):
        return messages, False

    logger.info("autocompact: threshold exceeded, compacting conversation...")

    try:
        compact_prompt = get_compact_prompt(custom_instructions)
        compact_messages = [
            {"role": "system", "content": compact_prompt},
            # Include the full conversation so the LLM can summarize it
            {"role": "user", "content": _serialize_conversation_for_compact(messages)},
        ]

        # Call LLM without tools (text-only mode)
        response = await provider.chat(
            messages=compact_messages,
            tools=None,
            model=model,
            max_tokens=MAX_OUTPUT_TOKENS_FOR_SUMMARY,
            temperature=0.0,
        )

        if not response.content:
            raise ValueError("Empty compact summary from LLM")

        # Build the replacement message
        summary_text = get_compact_user_summary_message(
            response.content,
            suppress_followup=True,
        )

        # Replace all messages with the compact summary
        new_messages = [
            {"role": "user", "content": summary_text},
        ]

        if tracking:
            tracking.compacted = True
            tracking.consecutive_failures = 0
            tracking.turn_counter = 0

        logger.info(
            f"autocompact: compressed {len(messages)} messages → 1 summary "
            f"(~{estimate_tokens_messages(new_messages)} tokens)"
        )

        return new_messages, True

    except Exception as e:
        logger.error(f"autocompact failed: {e}")
        if tracking:
            tracking.consecutive_failures += 1
        return messages, False


def _serialize_conversation_for_compact(messages: list[dict[str, Any]]) -> str:
    """Serialize conversation history for the compact LLM to summarize."""
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")

        if isinstance(content, list):
            # Multi-part content — extract text parts
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    t = part.get("text", "") or part.get("content", "")
                    if t:
                        text_parts.append(str(t))
            content = "\n".join(text_parts)

        # Include tool calls summary
        tool_calls = msg.get("tool_calls", [])
        tc_summary = ""
        if tool_calls:
            tc_names = []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function", {})
                name = fn.get("name", "?") if isinstance(fn, dict) else "?"
                tc_names.append(name)
            tc_summary = f"\n[Tool calls: {', '.join(tc_names)}]"

        parts.append(f"--- {role} ---\n{content}{tc_summary}")

    return "\n\n".join(parts)
