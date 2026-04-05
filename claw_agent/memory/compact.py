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
    session_persistence: Optional[Any] = None,  # SessionPersistence
) -> tuple[list[dict[str, Any]], bool]:
    """Run autocompact if needed. Returns (new_messages, was_compacted).
    Maps to: autoCompactIfNeeded() in autoCompact.ts

    This is the main entry point called by Engine after each turn.
    If session_persistence is provided and has non-empty notes, uses
    session-memory compact (trySessionMemoryCompaction). Otherwise
    falls back to legacy LLM compact.
    """
    if tracking and tracking.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
        logger.warning("autocompact: circuit breaker tripped, skipping")
        return messages, False

    if not should_auto_compact(messages, model):
        return messages, False

    logger.info("autocompact: threshold exceeded, compacting conversation...")

    # --- Try session-memory compact first ---
    # Maps to: trySessionMemoryCompaction() in sessionMemoryCompact.ts
    if session_persistence is not None:
        try:
            sm_result = await _try_session_memory_compact(
                messages, session_persistence, tracking
            )
            if sm_result is not None:
                return sm_result, True
        except Exception as e:
            logger.warning(f"autocompact: session-memory compact failed, falling back: {e}")

    # --- Fallback: legacy LLM compact ---
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


# ────────────────────────────────────────────────────────────────
# Session-memory compact — maps to sessionMemoryCompact.ts
# ────────────────────────────────────────────────────────────────

# Configuration for how many recent messages to preserve
SM_COMPACT_MIN_TOKENS = 10_000       # Minimum tokens to keep after compact
SM_COMPACT_MAX_TOKENS = 40_000       # Hard cap on preserved tokens
SM_COMPACT_MIN_TEXT_MESSAGES = 5     # Minimum messages with text blocks to keep


async def _try_session_memory_compact(
    messages: list[dict[str, Any]],
    session_persistence: Any,  # SessionPersistence
    tracking: Optional[CompactTrackingState] = None,
) -> Optional[list[dict[str, Any]]]:
    """Try to compact using session notes instead of LLM summarization.
    Maps to: trySessionMemoryCompaction() in sessionMemoryCompact.ts

    Returns new_messages on success, or None if session-memory compact
    cannot be used (falls back to legacy).
    """
    from claw_agent.memory.session_prompts import (
        is_session_notes_empty,
        truncate_session_notes_for_compact,
    )

    # Wait for any in-progress extraction to complete
    await session_persistence.wait_for_extraction()

    # Read session notes
    session_notes = session_persistence.get_content()
    if session_notes is None:
        logger.debug("autocompact: no session notes file, skipping SM compact")
        return None

    # Check if notes are still the blank template
    if is_session_notes_empty(session_notes):
        logger.debug("autocompact: session notes empty (template only), skipping SM compact")
        return None

    # Truncate oversized sections
    truncated_notes, was_truncated = truncate_session_notes_for_compact(session_notes)

    # Build summary from session notes
    summary_prefix = (
        "This session is being continued from a previous conversation that ran "
        "out of context. The session notes below were maintained during the earlier "
        "portion of the conversation and capture the key context.\n\n"
    )
    summary_text = summary_prefix + truncated_notes

    if was_truncated:
        summary_text += (
            f"\n\nSome session notes sections were truncated for length. "
            f"The full session notes can be viewed at: {session_persistence.notes_path}"
        )

    summary_text += (
        "\nContinue the conversation from where it left off without asking "
        "the user any further questions. Resume directly — do not acknowledge "
        "the summary, do not recap what was happening, do not preface with "
        '"I\'ll continue" or similar. Pick up the last task as if the break '
        "never happened."
    )

    # Determine which recent messages to keep (un-summarized)
    last_summarized_id = session_persistence.last_summarized_message_id
    keep_from_index = _calculate_messages_to_keep_index(
        messages, last_summarized_id
    )

    # Build final messages: [summary] + [recent un-summarized messages]
    recent_messages = messages[keep_from_index:]
    new_messages = [
        {"role": "user", "content": summary_text},
    ] + recent_messages

    if tracking:
        tracking.compacted = True
        tracking.consecutive_failures = 0
        tracking.turn_counter = 0

    logger.info(
        f"autocompact (session-memory): compressed {len(messages)} messages → "
        f"1 summary + {len(recent_messages)} recent "
        f"(~{estimate_tokens_messages(new_messages)} tokens)"
    )

    return new_messages


def _calculate_messages_to_keep_index(
    messages: list[dict[str, Any]],
    last_summarized_id: Optional[str],
) -> int:
    """Calculate the starting index for messages to keep after compact.
    Maps to: calculateMessagesToKeepIndex() in sessionMemoryCompact.ts

    Starts from last_summarized_id, then expands backwards to meet minimums.
    Finally adjusts the index to preserve tool_use/tool_result API invariants.
    """
    if not messages:
        return 0

    # Find the index of the last summarized message
    last_idx = len(messages)  # default: no messages summarized → keep none initially
    if last_summarized_id:
        for i, msg in enumerate(messages):
            msg_uuid = msg.get("uuid") or msg.get("id")
            if msg_uuid == last_summarized_id:
                last_idx = i + 1  # Start from message after the summarized one
                break

    start_index = last_idx

    # Calculate current token count and text-message count from start to end
    total_tokens = 0
    text_msg_count = 0
    for i in range(start_index, len(messages)):
        msg = messages[i]
        token_est = len(str(msg.get("content", ""))) // 4
        total_tokens += token_est
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            text_msg_count += 1

    # Check if we already meet both minimums
    if total_tokens >= SM_COMPACT_MIN_TOKENS and text_msg_count >= SM_COMPACT_MIN_TEXT_MESSAGES:
        return _adjust_index_to_preserve_api_invariants(messages, start_index)

    # Check if we already hit the max cap
    if total_tokens >= SM_COMPACT_MAX_TOKENS:
        return _adjust_index_to_preserve_api_invariants(messages, start_index)

    # Expand backwards until we meet both minimums or hit max cap
    for i in range(start_index - 1, -1, -1):
        msg = messages[i]
        token_est = len(str(msg.get("content", ""))) // 4
        total_tokens += token_est
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            text_msg_count += 1
        start_index = i

        if total_tokens >= SM_COMPACT_MAX_TOKENS:
            break
        if total_tokens >= SM_COMPACT_MIN_TOKENS and text_msg_count >= SM_COMPACT_MIN_TEXT_MESSAGES:
            break

    return _adjust_index_to_preserve_api_invariants(messages, start_index)


def _adjust_index_to_preserve_api_invariants(
    messages: list[dict[str, Any]],
    start_index: int,
) -> int:
    """Adjust the start index to ensure we don't split tool_use/tool_result pairs.
    Maps to: adjustIndexToPreserveAPIInvariants() in sessionMemoryCompact.ts

    If any message we're keeping contains tool_result blocks, we need to include
    the preceding assistant message(s) that contain the matching tool_use blocks.
    Otherwise the API will reject the request with an orphaned tool_result error.

    This handles two scenarios from the original TS:
      1. Tool pair splitting: a tool_result references a tool_use that would be pruned
      2. Shared message IDs: streaming yields separate messages per content block
         (thinking, tool_use) with the same message.id — if one is kept, all must be

    Example: if startIndex lands between an assistant[tool_use] and user[tool_result],
    we pull startIndex back to include the assistant message.
    """
    if start_index <= 0 or start_index >= len(messages):
        return start_index

    adjusted = start_index

    # ── Step 1: Collect all tool_result IDs from the kept range ──
    all_tool_result_ids: list[str] = []
    for i in range(start_index, len(messages)):
        msg = messages[i]
        if msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id")
            if tc_id:
                all_tool_result_ids.append(tc_id)

    if all_tool_result_ids:
        # Collect tool_use IDs already present in the kept range
        tool_use_ids_in_kept: set[str] = set()
        for i in range(adjusted, len(messages)):
            msg = messages[i]
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls", []):
                    tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    if tc_id:
                        tool_use_ids_in_kept.add(tc_id)

        # Find tool_result IDs whose matching tool_use is NOT in the kept range
        needed_ids = set(
            tid for tid in all_tool_result_ids
            if tid not in tool_use_ids_in_kept
        )

        # Walk backwards to find assistant messages with the missing tool_use blocks
        for i in range(adjusted - 1, -1, -1):
            if not needed_ids:
                break
            msg = messages[i]
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls", []):
                    tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    if tc_id and tc_id in needed_ids:
                        adjusted = i
                        needed_ids.discard(tc_id)

    # ── Step 2: Handle shared message IDs (thinking blocks) ──
    # Collect all message IDs from assistant messages in the kept range
    msg_ids_in_kept: set[str] = set()
    for i in range(adjusted, len(messages)):
        msg = messages[i]
        if msg.get("role") == "assistant":
            mid = msg.get("message_id") or msg.get("id")
            if mid:
                msg_ids_in_kept.add(mid)

    # Walk backwards for assistant messages sharing a message_id with kept ones
    if msg_ids_in_kept:
        for i in range(adjusted - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") == "assistant":
                mid = msg.get("message_id") or msg.get("id")
                if mid and mid in msg_ids_in_kept:
                    adjusted = i

    return adjusted


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
