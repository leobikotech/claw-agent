"""
Dream — 后台记忆巩固引擎
Maps to: src/services/autoDream/autoDream.ts, consolidationPrompt.ts, consolidationLock.ts

Background memory consolidation with three-gate trigger + four-phase pipeline.
三门触发 + 四阶段记忆整合的后台巩固引擎。

Gate order (cheapest first) — 与原版一致:
  1. Time: hours since lastConsolidatedAt >= minHours  (one stat)
  2. Sessions: session_count >= minSessions
  3. Lock: no other process mid-consolidation (file-based PID lock)
"""
from __future__ import annotations
import os
import time
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from claw_agent.memory.memory import Memory
    from claw_agent.config import Config

logger = logging.getLogger(__name__)

# Scan throttle: when time-gate passes but session-gate doesn't,
# avoid re-scanning every turn.
# Maps to: SESSION_SCAN_INTERVAL_MS in autoDream.ts
SESSION_SCAN_INTERVAL_S = 10 * 60  # 10 minutes

# Stale lock threshold — even if PID is live, reclaim after this.
# Maps to: HOLDER_STALE_MS in consolidationLock.ts
LOCK_STALE_S = 60 * 60  # 1 hour


@dataclass
class DreamConfig:
    """Maps to: AutoDreamConfig in autoDream.ts"""
    min_hours: float = 24.0     # Time gate threshold
    min_sessions: int = 5       # Session gate threshold


@dataclass
class DreamState:
    """In-memory state (augmented by file lock for cross-process safety)"""
    last_consolidated_at: float = 0.0
    session_count: int = 0


# ────────────────────────────────────────────────────────────────
# File-based cross-process lock
# Maps to: src/services/autoDream/consolidationLock.ts
# ────────────────────────────────────────────────────────────────

LOCK_FILE = ".consolidate-lock"


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


class ConsolidationLock:
    """File-based cross-process consolidation lock / 跨进程巩固锁
    Maps to: consolidationLock.ts

    Lock file body = holder PID
    Lock file mtime = lastConsolidatedAt

    Acquire: write PID → mtime = now
    Success: mtime stays at now (consolidation completed)
    Failure: rollback(priorMtime) rewinds mtime
    Crash:   mtime stuck, dead PID → next process reclaims
    """

    def __init__(self, memory_dir: str):
        self.lock_path = os.path.join(memory_dir, LOCK_FILE)
        self.memory_dir = memory_dir

    def read_last_consolidated_at(self) -> float:
        """mtime of the lock file = lastConsolidatedAt. 0 if absent.
        Per-turn cost: one stat.
        Maps to: readLastConsolidatedAt() in consolidationLock.ts
        """
        try:
            return os.stat(self.lock_path).st_mtime
        except FileNotFoundError:
            return 0.0

    def try_acquire(self) -> Optional[float]:
        """Acquire the lock. Returns pre-acquire mtime (for rollback), or None if blocked.
        Maps to: tryAcquireConsolidationLock() in consolidationLock.ts
        """
        prior_mtime: Optional[float] = None
        holder_pid: Optional[int] = None

        try:
            stat = os.stat(self.lock_path)
            prior_mtime = stat.st_mtime
            with open(self.lock_path, "r") as f:
                raw = f.read().strip()
            if raw.isdigit():
                holder_pid = int(raw)
        except FileNotFoundError:
            pass  # No prior lock

        # Check if lock is held by a live process within stale threshold
        if prior_mtime is not None:
            age_s = time.time() - prior_mtime
            if age_s < LOCK_STALE_S:
                if holder_pid is not None and _is_pid_alive(holder_pid):
                    logger.debug(
                        f"[dream] lock held by live PID {holder_pid} "
                        f"(mtime {int(age_s)}s ago)"
                    )
                    return None
                # Dead PID or unparseable body — reclaim

        # Write our PID
        os.makedirs(self.memory_dir, exist_ok=True)
        my_pid = os.getpid()
        with open(self.lock_path, "w") as f:
            f.write(str(my_pid))

        # Verify we won the race (two reclaimers both write → last wins)
        try:
            with open(self.lock_path, "r") as f:
                verify = f.read().strip()
            if verify != str(my_pid):
                return None
        except FileNotFoundError:
            return None

        return prior_mtime if prior_mtime is not None else 0.0

    def rollback(self, prior_mtime: float):
        """Rewind mtime to pre-acquire after a failed consolidation.
        Maps to: rollbackConsolidationLock() in consolidationLock.ts
        """
        try:
            if prior_mtime == 0.0:
                os.unlink(self.lock_path)
                return
            # Clear PID body so our still-running process doesn't look like it's holding
            with open(self.lock_path, "w") as f:
                f.write("")
            os.utime(self.lock_path, (prior_mtime, prior_mtime))
        except Exception as e:
            logger.error(
                f"[dream] rollback failed: {e} — next trigger delayed to minHours"
            )

    def release(self):
        """Mark consolidation as complete (mtime stays at now)"""
        try:
            # Clear PID body but leave the file (mtime = completion timestamp)
            with open(self.lock_path, "w") as f:
                f.write("")
        except Exception as e:
            logger.error(f"[dream] release failed: {e}")


# ────────────────────────────────────────────────────────────────
# Consolidation Prompt
# Maps to: src/services/autoDream/consolidationPrompt.ts
# ────────────────────────────────────────────────────────────────

def build_consolidation_prompt(memory_dir: str, memory_file: str = "MEMORY.md") -> str:
    """Build the four-phase consolidation prompt / 构建四阶段巩固提示词
    Maps to: buildConsolidationPrompt() in consolidationPrompt.ts

    Preserves ALL phases from the original:
    1. Orient — read existing memory index
    2. Gather recent signal — check topic files and daily logs
    3. Consolidate — merge, deduplicate, fix dates
    4. Prune and index — keep MEMORY.md concise
    """
    return f"""# Dream: Memory Consolidation

You are performing a dream — a reflective pass over your memory files. Synthesize what you've learned recently into durable, well-organized memories so that future sessions can orient quickly.

Memory directory: `{memory_dir}`
If the directory doesn't exist yet, create it before writing any files.

---

## Phase 1 — Orient

- `ls` the memory directory to see what already exists
- Read `{memory_file}` to understand the current index
- Skim existing topic files so you improve them rather than creating duplicates
- If `logs/` or `sessions/` subdirectories exist, review recent entries there

## Phase 2 — Gather recent signal

Look for new information worth persisting. Sources in rough priority order:

1. **Daily logs** (`logs/YYYY/MM/YYYY-MM-DD.md`) if present — these are the append-only stream
2. **Existing memories that drifted** — facts that contradict something you see in the codebase now
3. **Topic files** — check if any need updating based on recent work

Don't exhaustively read everything. Look only for things you already suspect matter.

## Phase 3 — Consolidate

For each thing worth remembering, write or update a memory file at the top level of the memory directory.

Focus on:
- Merging new signal into existing topic files rather than creating near-duplicates
- Converting relative dates ("yesterday", "last week") to absolute dates so they remain interpretable after time passes
- Deleting contradicted facts — if today's investigation disproves an old memory, fix it at the source
- Ensuring consistency across all entries

## Phase 4 — Prune and index

Update `{memory_file}` so it stays under 200 lines AND under ~25KB. It's an **index**, not a dump — each entry should be one line under ~150 characters: `- [Title](file.md) — one-line hook`. Never write memory content directly into it.

- Remove pointers to memories that are now stale, wrong, or superseded
- Demote verbose entries: if an index line is over ~200 chars, it's carrying content that belongs in the topic file — shorten the line, move the detail
- Add pointers to newly important memories
- Resolve contradictions — if two files disagree, fix the wrong one

---

**Tool constraints for this run:** Bash is restricted to read-only commands (`ls`, `find`, `grep`, `cat`, `stat`, `wc`, `head`, `tail`, and similar). Use the dedicated file_write and file_edit tools to modify files. Anything that writes via bash will be denied.

Important: Actually apply the file changes using your tools. Do not just output what you would do.

Return a brief summary of what you consolidated, updated, or pruned. If nothing changed (memories are already tight), say so."""


# ────────────────────────────────────────────────────────────────
# DreamEngine — 后台记忆巩固引擎
# Maps to: initAutoDream() + executeAutoDream() in autoDream.ts
# ────────────────────────────────────────────────────────────────

class DreamEngine:
    """Background memory consolidation / 后台记忆巩固"""

    def __init__(
        self,
        memory: Memory,
        app_config: Config,
        dream_config: Optional[DreamConfig] = None,
        parent_engine: Optional[Any] = None,
    ):
        self.memory = memory
        self.app_config = app_config
        self.dream_config = dream_config or DreamConfig()
        self.lock = ConsolidationLock(memory.memory_dir)
        self._state_path = os.path.join(memory.memory_dir, ".dream_state.json")
        self._last_scan_at: float = 0.0  # Scan throttle timestamp
        # Optional reference to the parent engine for abort propagation
        self._parent_engine = parent_engine
        self.state = self._load_state()

    def _load_state(self) -> DreamState:
        if os.path.exists(self._state_path):
            try:
                with open(self._state_path) as f:
                    data = json.load(f)
                return DreamState(**{
                    k: data[k] for k in ("last_consolidated_at", "session_count")
                    if k in data
                })
            except Exception:
                pass
        return DreamState()

    def _save_state(self):
        try:
            with open(self._state_path, "w") as f:
                json.dump({
                    "last_consolidated_at": self.state.last_consolidated_at,
                    "session_count": self.state.session_count,
                }, f)
        except Exception as e:
            logger.error(f"Failed to save dream state: {e}")

    def record_session(self):
        """Record a session completion / 记录一次会话完成"""
        self.state.session_count += 1
        self._save_state()

    def should_dream(self) -> tuple[bool, str]:
        """Three-gate check / 三门触发检查
        Maps to: Gate order in autoDream.ts (cheapest first)
        """
        # Gate 1: Time
        last_at = self.lock.read_last_consolidated_at()
        if last_at > 0:
            hours_since = (time.time() - last_at) / 3600
        else:
            hours_since = float("inf")  # Never consolidated → always pass time gate

        if hours_since < self.dream_config.min_hours:
            return False, f"Time gate: {hours_since:.1f}h < {self.dream_config.min_hours}h"

        # Scan throttle
        since_scan = time.time() - self._last_scan_at
        if since_scan < SESSION_SCAN_INTERVAL_S:
            return False, (
                f"Scan throttle: last scan was {int(since_scan)}s ago "
                f"(need {SESSION_SCAN_INTERVAL_S}s)"
            )
        self._last_scan_at = time.time()

        # Gate 2: Sessions
        if self.state.session_count < self.dream_config.min_sessions:
            return False, (
                f"Session gate: {self.state.session_count} < {self.dream_config.min_sessions}"
            )

        return True, "All gates passed"

    async def run(self) -> str:
        """Execute the dream consolidation / 执行记忆巩固
        Maps to: executeAutoDream() in autoDream.ts

        Acquires file lock → spins up lightweight Engine → runs consolidation → releases lock
        """
        should, reason = self.should_dream()
        if not should:
            return f"Dream skipped: {reason}"

        # Gate 3: File-based lock
        prior_mtime = self.lock.try_acquire()
        if prior_mtime is None:
            return "Dream skipped: Lock gate — consolidation already in progress"

        try:
            logger.info("💤 Dream consolidation started...")

            # Import Engine + tools dynamically to avoid circular dependencies
            from claw_agent.core.engine import Engine
            from claw_agent.tools.bash_tool import BashTool
            from claw_agent.tools.file_tools import FileReadTool, FileWriteTool, FileEditTool

            # Provide file tools for the dream agent (not just BashTool)
            from claw_agent.memory.memory import MEMORY_FILE
            tools = [BashTool(), FileReadTool(), FileWriteTool(), FileEditTool()]
            engine = Engine(config=self.app_config, tools=tools)

            prompt = build_consolidation_prompt(self.memory.memory_dir, MEMORY_FILE)

            # Run the engine headless
            output = ""
            async for event in engine.run(prompt, max_turns=10):
                # Bug 5 fix: propagate parent abort into dream sub-engine
                if self._parent_engine and self._parent_engine.aborted:
                    engine.abort()
                    output = "Dream aborted: parent engine was cancelled"
                    break
                if event["type"] in ("done", "error"):
                    output = event["content"]

            # Success: update state
            self.state.last_consolidated_at = time.time()
            self.state.session_count = 0
            self._save_state()
            self.lock.release()

            logger.info("💤 Dream consolidation completed.")
            return f"Dream success: {output}"

        except Exception as e:
            # Rollback lock mtime so time-gate passes again on next attempt
            self.lock.rollback(prior_mtime)
            logger.error(f"Dream failed: {e}")
            return f"Dream failed: {e}"

    async def run_force(self) -> str:
        """Force-run dream consolidation, bypassing all gates.
        强制执行记忆巩固，跳过所有门控。
        Maps to: manual /dream command in commands/dream.ts
        """
        prior_mtime = self.lock.try_acquire()
        if prior_mtime is None:
            return "Dream skipped: Lock gate — consolidation already in progress"

        try:
            logger.info("💤 Dream consolidation started (forced)...")

            from claw_agent.core.engine import Engine
            from claw_agent.tools.bash_tool import BashTool
            from claw_agent.tools.file_tools import FileReadTool, FileWriteTool, FileEditTool
            from claw_agent.memory.memory import MEMORY_FILE

            tools = [BashTool(), FileReadTool(), FileWriteTool(), FileEditTool()]
            engine = Engine(config=self.app_config, tools=tools)

            prompt = build_consolidation_prompt(self.memory.memory_dir, MEMORY_FILE)

            output = ""
            async for event in engine.run(prompt, max_turns=10):
                if self._parent_engine and self._parent_engine.aborted:
                    engine.abort()
                    output = "Dream aborted: parent engine was cancelled"
                    break
                if event["type"] in ("done", "error"):
                    output = event["content"]

            self.state.last_consolidated_at = time.time()
            self.state.session_count = 0
            self._save_state()
            self.lock.release()

            logger.info("💤 Dream consolidation completed (forced).")
            return f"Dream success: {output}"

        except Exception as e:
            self.lock.rollback(prior_mtime)
            logger.error(f"Dream failed: {e}")
            return f"Dream failed: {e}"


# ────────────────────────────────────────────────────────────────
# Hook factories — create HookCallbacks for registration
# Maps to: executeAutoDream() in stopHooks.ts:155
# ────────────────────────────────────────────────────────────────

def create_dream_hook(dream_engine: DreamEngine):
    """Create a fire-and-forget HookCallback that triggers dream consolidation.
    创建一个触发记忆巩固的后台钩子回调。

    Maps to: `void executeAutoDream(stopHookContext, ...)` in stopHooks.ts:155
    Registered as STOP event hook with fire_and_forget=True.

    The three-gate check (time, sessions, lock) inside DreamEngine.run()
    makes this cheap per-turn — only one stat + one counter check.
    """
    from claw_agent.core.hooks import HookContext, HookResult

    async def _dream_hook(ctx: HookContext) -> Optional[HookResult]:
        result = await dream_engine.run()
        if "success" in result.lower():
            return HookResult(additional_context=result)
        logger.debug(f"[dream_hook] {result}")
        return None

    _dream_hook.__name__ = "auto_dream"
    return _dream_hook


def create_session_record_hook(dream_engine: DreamEngine):
    """Create a SESSION_END hook that increments the session counter.
    创建一个 SESSION_END 钩子，递增会话计数器。

    Maps to: session counting logic that feeds Gate 2 (min_sessions)
    in autoDream.ts. Original tracks via listSessionsTouchedSince();
    we simplify to an explicit counter.
    """
    from claw_agent.core.hooks import HookContext, HookResult

    async def _session_record_hook(ctx: HookContext) -> Optional[HookResult]:
        dream_engine.record_session()
        logger.debug(
            f"[session_record] Session count: {dream_engine.state.session_count}"
        )
        return None

    _session_record_hook.__name__ = "session_record"
    return _session_record_hook
