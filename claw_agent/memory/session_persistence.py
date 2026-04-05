"""
Session Persistence — 会话持久化系统
Maps to: src/services/SessionMemory/sessionMemory.ts + sessionMemoryUtils.ts

Automatically maintains a structured markdown file with notes about the current
conversation. Runs periodically via a forked sub-agent to extract key information
without interrupting the main conversation flow.

Architecture:
  1. Engine fires POST_SAMPLING hook after each LLM response
  2. SessionPersistence checks dual-threshold (tokens + tool calls)
  3. If triggered → forked Engine instance runs with restricted tools
  4. Sub-agent edits the session notes file at ~/.claw/sessions/<cwd_hash>/session-notes.md
  5. On compact, session notes replace the legacy LLM-generated summary
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from claw_agent.memory.session_prompts import (
    DEFAULT_SESSION_NOTES_TEMPLATE,
    build_session_update_prompt,
    is_session_notes_empty,
    truncate_session_notes_for_compact,
)

if TYPE_CHECKING:
    from claw_agent.config import Config

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────
# Constants — matching sessionMemoryUtils.ts
# ────────────────────────────────────────────────────────────────

SESSION_NOTES_FILENAME = "session-notes.md"
EXTRACTION_WAIT_TIMEOUT_S = 15
EXTRACTION_STALE_THRESHOLD_S = 60  # 1 minute


# ────────────────────────────────────────────────────────────────
# Configuration — maps to SessionMemoryConfig in sessionMemoryUtils.ts
# ────────────────────────────────────────────────────────────────

@dataclass
class SessionPersistenceConfig:
    """Threshold configuration for session persistence extraction.
    Maps to: SessionMemoryConfig in sessionMemoryUtils.ts

    Attributes:
        min_tokens_to_init: Minimum context window tokens before first extraction.
        min_tokens_between_update: Minimum token growth between extractions.
        tool_calls_between_updates: Minimum tool calls between extractions.
    """
    min_tokens_to_init: int = 10_000
    min_tokens_between_update: int = 5_000
    tool_calls_between_updates: int = 3


# ────────────────────────────────────────────────────────────────
# Main class — maps to sessionMemory.ts + sessionMemoryUtils.ts
# ────────────────────────────────────────────────────────────────

class SessionPersistence:
    """Session persistence — maintains structured notes across conversations.
    会话持久化——跨对话维护结构化笔记。

    Maps to:
      - initSessionMemory() / extractSessionMemory() in sessionMemory.ts
      - SessionMemoryConfig / state functions in sessionMemoryUtils.ts
    """

    def __init__(
        self,
        cwd: str,
        config: Optional[SessionPersistenceConfig] = None,
    ):
        self.cwd = cwd
        self.config = config or SessionPersistenceConfig()

        # --- Internal state (maps to sessionMemoryUtils.ts module variables) ---
        self._initialized = False           # Has met min_tokens_to_init threshold
        self._tokens_at_last_extraction = 0 # Token count at last extraction
        self._last_extraction_message_uuid: Optional[str] = None  # UUID of last message at extraction time
        self._last_summarized_message_id: Optional[str] = None    # For compact integration
        self._extraction_started_at: Optional[float] = None       # Timestamp of in-progress extraction
        self._extract_lock = asyncio.Lock()  # Ensure sequential extraction

        # --- File paths ---
        self._session_dir = self._compute_session_dir()
        self._notes_path = os.path.join(self._session_dir, SESSION_NOTES_FILENAME)

    # ────────────────────────────────────────────────────────────
    # Path computation
    # ────────────────────────────────────────────────────────────

    def _compute_session_dir(self) -> str:
        """Compute session directory: ~/.claw/sessions/<cwd_hash>/"""
        cwd_hash = hashlib.sha256(self.cwd.encode()).hexdigest()[:12]
        home = os.path.expanduser("~")
        return os.path.join(home, ".claw", "sessions", cwd_hash)

    @property
    def session_dir(self) -> str:
        """The session storage directory path."""
        return self._session_dir

    @property
    def notes_path(self) -> str:
        """The session notes file path."""
        return self._notes_path

    @property
    def last_summarized_message_id(self) -> Optional[str]:
        """The message ID up to which session notes are current.
        Used by compact to determine which messages to keep.
        Maps to: getLastSummarizedMessageId() in sessionMemoryUtils.ts
        """
        return self._last_summarized_message_id

    # ────────────────────────────────────────────────────────────
    # Threshold checks — maps to shouldExtractMemory()
    # ────────────────────────────────────────────────────────────

    def _estimate_token_count(self, messages: list) -> int:
        """Rough token estimation for messages.
        Uses the same approach as compact — estimate from serialized content.
        """
        from claw_agent.providers.tokens import estimate_tokens_messages
        from claw_agent.core.messages import messages_to_api
        api_msgs = messages_to_api(messages)
        return estimate_tokens_messages(api_msgs)

    def _count_tool_calls_since(
        self,
        messages: list,
        since_uuid: Optional[str],
    ) -> int:
        """Count tool calls in assistant messages since a given message UUID.
        Maps to: countToolCallsSince() in sessionMemory.ts
        """
        tool_call_count = 0
        found_start = since_uuid is None

        for msg in messages:
            if not found_start:
                if hasattr(msg, "uuid") and msg.uuid == since_uuid:
                    found_start = True
                elif hasattr(msg, "tool_calls"):
                    # For messages without uuid, check by identity
                    pass
                continue

            # Count tool_calls in assistant messages
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_call_count += len(msg.tool_calls)

        return tool_call_count

    def _has_tool_calls_in_last_assistant_turn(self, messages: list) -> bool:
        """Check if the last assistant message has tool calls.
        Maps to: hasToolCallsInLastAssistantTurn() in utils/messages.ts
        """
        for msg in reversed(messages):
            if hasattr(msg, "tool_calls"):
                return bool(msg.tool_calls)
        return False

    def should_extract(self, messages: list) -> bool:
        """Check if session notes should be extracted now.
        Maps to: shouldExtractMemory() in sessionMemory.ts

        Dual-threshold logic:
          1. Token threshold: context has grown by ≥ min_tokens_between_update
          2. Tool call threshold: ≥ tool_calls_between_updates OR no tool calls in last turn

        Both token growth AND tool-call / conversation-break conditions must be met.
        """
        current_token_count = self._estimate_token_count(messages)

        # Gate: initialization threshold
        if not self._initialized:
            if current_token_count < self.config.min_tokens_to_init:
                return False
            self._initialized = True
            logger.debug(
                f"[session_persistence] Initialized at {current_token_count} tokens"
            )

        # Gate: token growth since last extraction
        token_growth = current_token_count - self._tokens_at_last_extraction
        has_met_token_threshold = (
            token_growth >= self.config.min_tokens_between_update
        )

        # Gate: tool call count since last extraction
        tool_calls_since = self._count_tool_calls_since(
            messages, self._last_extraction_message_uuid
        )
        has_met_tool_call_threshold = (
            tool_calls_since >= self.config.tool_calls_between_updates
        )

        # Check if last assistant turn has no tool calls (natural break)
        has_tool_calls_in_last_turn = self._has_tool_calls_in_last_assistant_turn(
            messages
        )

        # Trigger when:
        # 1. Token threshold met AND tool call threshold met, OR
        # 2. Token threshold met AND no tool calls in last turn (natural break)
        should = (
            (has_met_token_threshold and has_met_tool_call_threshold)
            or (has_met_token_threshold and not has_tool_calls_in_last_turn)
        )

        if should:
            # Record the last message for next check
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "uuid"):
                    self._last_extraction_message_uuid = last_msg.uuid

        return should

    # ────────────────────────────────────────────────────────────
    # File operations
    # ────────────────────────────────────────────────────────────

    def _ensure_session_dir(self) -> None:
        """Create session directory if it doesn't exist."""
        os.makedirs(self._session_dir, mode=0o700, exist_ok=True)

    def _ensure_notes_file(self) -> None:
        """Create session notes file with template if it doesn't exist.
        Maps to: setupSessionMemoryFile() in sessionMemory.ts
        """
        self._ensure_session_dir()
        if not os.path.exists(self._notes_path):
            with open(self._notes_path, "w", encoding="utf-8") as f:
                f.write(DEFAULT_SESSION_NOTES_TEMPLATE)
            os.chmod(self._notes_path, 0o600)

    def _read_notes(self) -> str:
        """Read current session notes content."""
        self._ensure_notes_file()
        with open(self._notes_path, "r", encoding="utf-8") as f:
            return f.read()

    def get_content(self) -> Optional[str]:
        """Get current session notes content, or None if no file exists.
        Maps to: getSessionMemoryContent() in sessionMemoryUtils.ts
        """
        try:
            if not os.path.exists(self._notes_path):
                return None
            with open(self._notes_path, "r", encoding="utf-8") as f:
                content = f.read()
            return content
        except OSError:
            return None

    # ────────────────────────────────────────────────────────────
    # Extraction — maps to extractSessionMemory() in sessionMemory.ts
    # ────────────────────────────────────────────────────────────

    async def extract(
        self,
        messages: list,
        app_config: "Config",
    ) -> None:
        """Run session notes extraction via a forked sub-agent.
        Maps to: extractSessionMemory() in sessionMemory.ts

        Creates a lightweight Engine instance restricted to file-edit-only,
        passes the full conversation as context, and instructs it to update
        the session notes markdown file.
        """
        async with self._extract_lock:
            self._extraction_started_at = time.time()

            try:
                # Read current notes
                current_notes = self._read_notes()
                notes_path = self._notes_path

                # Build the extraction prompt
                user_prompt = build_session_update_prompt(current_notes, notes_path)

                # Import Engine + tools dynamically to avoid circular deps
                from claw_agent.core.engine import Engine
                from claw_agent.core.messages import messages_to_api
                from claw_agent.tools.file_tools import FileReadTool, FileEditTool
                import os

                class RestrictedFileEditTool(FileEditTool):
                    """A sandboxed file_edit tool strictly limited to the session notes file."""
                    def __init__(self, allowed_path: str):
                        super().__init__()
                        self.allowed_path = os.path.abspath(allowed_path)
                        self.description = f"Overridden file_edit tool. You may ONLY edit {self.allowed_path}."

                    def execute(self, **kwargs) -> Any:
                        path_arg = kwargs.get("file_path", "")
                        if not path_arg or os.path.abspath(path_arg) != self.allowed_path:
                            return f"Error: Permission denied. You are only allowed to edit exactly {self.allowed_path}."
                        return super().execute(**kwargs)

                # Create restricted tool set (only file read + sandboxed edit)
                tools = [FileReadTool(), RestrictedFileEditTool(notes_path)]
                engine = Engine(config=app_config, tools=tools)

                # Seed the forked engine with the full conversation context
                # so the extraction agent can see everything discussed
                engine.messages = list(messages)

                # Run the extraction — the sub-agent edits the notes file
                async for event in engine.run(user_prompt, max_turns=5):
                    if event["type"] == "error":
                        logger.error(
                            f"[session_persistence] Extraction error: {event['content']}"
                        )
                        break
                    elif event["type"] == "done":
                        break

                # Record extraction state
                current_token_count = self._estimate_token_count(messages)
                self._tokens_at_last_extraction = current_token_count

                # Update last_summarized_message_id if safe
                if not self._has_tool_calls_in_last_assistant_turn(messages):
                    if messages:
                        last_msg = messages[-1]
                        if hasattr(last_msg, "uuid"):
                            self._last_summarized_message_id = last_msg.uuid

                logger.info("[session_persistence] Extraction completed successfully")

            except Exception as e:
                logger.error(f"[session_persistence] Extraction failed: {e}")

            finally:
                self._extraction_started_at = None

    # ────────────────────────────────────────────────────────────
    # Extraction wait — maps to waitForSessionMemoryExtraction()
    # ────────────────────────────────────────────────────────────

    async def wait_for_extraction(self) -> None:
        """Wait for any in-progress extraction to complete (with timeout).
        Maps to: waitForSessionMemoryExtraction() in sessionMemoryUtils.ts

        Returns immediately if no extraction is running or if extraction is stale (>1min).
        """
        start = time.time()
        while self._extraction_started_at is not None:
            age = time.time() - self._extraction_started_at
            if age > EXTRACTION_STALE_THRESHOLD_S:
                # Stale extraction — don't wait
                return

            if time.time() - start > EXTRACTION_WAIT_TIMEOUT_S:
                # Timeout
                return

            await asyncio.sleep(1.0)

    # ────────────────────────────────────────────────────────────
    # Hook integration — called by Engine via POST_SAMPLING hook
    # ────────────────────────────────────────────────────────────

    async def maybe_extract(
        self,
        messages: list,
        app_config: "Config",
    ) -> None:
        """Check thresholds and run extraction if needed.
        Called by the POST_SAMPLING hook (fire-and-forget).
        """
        if not self.should_extract(messages):
            return

        await self.extract(messages, app_config)

    # ────────────────────────────────────────────────────────────
    # Reset — for testing
    # ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all internal state.
        Maps to: resetSessionMemoryState() in sessionMemoryUtils.ts
        """
        self._initialized = False
        self._tokens_at_last_extraction = 0
        self._last_extraction_message_uuid = None
        self._last_summarized_message_id = None
        self._extraction_started_at = None


# ────────────────────────────────────────────────────────────────
# Hook factory — create HookCallback for Engine registration
# Maps to: registerPostSamplingHook(extractSessionMemory) in sessionMemory.ts
# ────────────────────────────────────────────────────────────────

def create_session_persistence_hook(
    session_persistence: SessionPersistence,
    app_config: "Config",
):
    """Create a fire-and-forget hook callback for session persistence extraction.
    创建会话持久化提取的后台钩子回调。

    Maps to: registerPostSamplingHook(extractSessionMemory) in sessionMemory.ts
    Registered as POST_SAMPLING event hook with fire_and_forget=True.
    """
    from claw_agent.core.hooks import HookContext, HookResult

    async def _session_persistence_hook(ctx: HookContext) -> Optional[HookResult]:
        await session_persistence.maybe_extract(ctx.messages, app_config)
        return None

    _session_persistence_hook.__name__ = "session_persistence"
    return _session_persistence_hook
