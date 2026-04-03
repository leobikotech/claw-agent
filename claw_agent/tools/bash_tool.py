"""
BashTool — Shell 执行工具
Maps to: src/tools/BashTool/BashTool.tsx + prompt.ts (370 lines)

Execute shell commands in a subprocess with timeout, background execution,
and comprehensive safety/usage instructions matching the original prompt.
"""
from __future__ import annotations
import asyncio
import os
from typing import Any

from claw_agent.core.tool import Tool, ToolContext, RiskLevel, PermissionCheck
from claw_agent.core.permissions import is_dangerous_command

# Constants matching original toolName.ts
BASH_TOOL_NAME = "bash"
DEFAULT_TIMEOUT_S = 120
MAX_TIMEOUT_S = 600

# Tool name refs for prompt cross-references
FILE_READ_TOOL_NAME = "file_read"
FILE_EDIT_TOOL_NAME = "file_edit"
FILE_WRITE_TOOL_NAME = "file_write"
GLOB_TOOL_NAME = "glob"
GREP_TOOL_NAME = "grep"
AGENT_TOOL_NAME = "agent"


def _get_bash_prompt() -> str:
    """Build the full BashTool prompt — 1:1 translation of prompt.ts getSimplePrompt().

    This is the most critical prompt in the system: it governs when the LLM
    uses bash vs dedicated tools, how it handles git, and how it chains commands.
    """
    tool_preference_items = [
        f"File search: Use {GLOB_TOOL_NAME} (NOT find or ls)",
        f"Content search: Use {GREP_TOOL_NAME} (NOT grep or rg)",
        f"Read files: Use {FILE_READ_TOOL_NAME} (NOT cat/head/tail)",
        f"Edit files: Use {FILE_EDIT_TOOL_NAME} (NOT sed/awk)",
        f"Write files: Use {FILE_WRITE_TOOL_NAME} (NOT echo >/cat <<EOF)",
        "Communication: Output text directly (NOT echo/printf)",
    ]

    avoid_commands = "`find`, `grep`, `cat`, `head`, `tail`, `sed`, `awk`, or `echo`"

    multiple_commands_subitems = [
        f"If the commands are independent and can run in parallel, make multiple {BASH_TOOL_NAME} tool calls in a single message. Example: if you need to run \"git status\" and \"git diff\", send a single message with two {BASH_TOOL_NAME} tool calls in parallel.",
        f"If the commands depend on each other and must run sequentially, use a single {BASH_TOOL_NAME} call with '&&' to chain them together.",
        "Use ';' only when you need to run commands sequentially but don't care if earlier commands fail.",
        "DO NOT use newlines to separate commands (newlines are ok in quoted strings).",
    ]

    git_subitems = [
        "Prefer to create a new commit rather than amending an existing commit.",
        "Before running destructive operations (e.g., git reset --hard, git push --force, git checkout --), consider whether there is a safer alternative that achieves the same goal. Only use destructive operations when they are truly the best approach.",
        "Never skip hooks (--no-verify) or bypass signing (--no-gpg-sign, -c commit.gpgsign=false) unless the user has explicitly asked for it. If a hook fails, investigate and fix the underlying issue.",
    ]

    sleep_subitems = [
        "Do not sleep between commands that can run immediately — just run them.",
        "If your command is long running and you would like to be notified when it finishes — use `run_in_background`. No sleep needed.",
        "Do not retry failing commands in a sleep loop — diagnose the root cause.",
        "If waiting for a background task you started with `run_in_background`, you will be notified when it completes — do not poll.",
        "If you must poll an external process, use a check command (e.g. `gh run view`) rather than sleeping first.",
        "If you must sleep, keep the duration short (1-5 seconds) to avoid blocking the user.",
    ]

    background_note = (
        "You can use the `run_in_background` parameter to run the command in the "
        "background. Only use this if you don't need the result immediately and are "
        "OK being notified when the command completes later. You do not need to check "
        "the output right away - you'll be notified when it finishes. You do not need "
        "to use '&' at the end of the command when using this parameter."
    )

    instruction_items = [
        "If your command will create new directories or files, first use this tool to run `ls` to verify the parent directory exists and is the correct location.",
        "Always quote file paths that contain spaces with double quotes in your command (e.g., cd \"path with spaces/file.txt\")",
        "Try to maintain your current working directory throughout the session by using absolute paths and avoiding usage of `cd`. You may use `cd` if the User explicitly requests it.",
        f"You may specify an optional timeout in seconds (up to {MAX_TIMEOUT_S}s / {MAX_TIMEOUT_S // 60} minutes). By default, your command will timeout after {DEFAULT_TIMEOUT_S}s ({DEFAULT_TIMEOUT_S // 60} minutes).",
        background_note,
    ]

    # --- Git commit/PR instructions (full translation of getCommitAndPRInstructions) ---
    git_instructions = _get_commit_and_pr_instructions()

    # --- Assemble final prompt ---
    pref_bullets = "\n".join(f"- {item}" for item in tool_preference_items)
    instr_bullets = "\n".join(f"- {item}" for item in instruction_items)
    multi_cmd_bullets = "\n".join(f"  - {item}" for item in multiple_commands_subitems)
    git_cmd_bullets = "\n".join(f"  - {item}" for item in git_subitems)
    sleep_bullets = "\n".join(f"  - {item}" for item in sleep_subitems)

    return f"""Executes a given bash command and returns its output.

The working directory persists between commands, but shell state does not. The shell environment is initialized from the user's profile (bash or zsh).

IMPORTANT: Avoid using this tool to run {avoid_commands} commands, unless explicitly instructed or after you have verified that a dedicated tool cannot accomplish your task. Instead, use the appropriate dedicated tool as this will provide a much better experience for the user:

{pref_bullets}
While the {BASH_TOOL_NAME} tool can do similar things, it's better to use the built-in tools as they provide a better user experience and make it easier to review tool calls and give permission.

# Instructions
{instr_bullets}
- When issuing multiple commands:
{multi_cmd_bullets}
- For git commands:
{git_cmd_bullets}
- Avoid unnecessary `sleep` commands:
{sleep_bullets}
{git_instructions}"""


def _get_commit_and_pr_instructions() -> str:
    """Full translation of getCommitAndPRInstructions() from prompt.ts."""
    return f"""
# Committing changes with git

Only create commits when requested by the user. If unclear, ask first. When the user asks you to create a new git commit, follow these steps carefully:

You can call multiple tools in a single response. When multiple independent pieces of information are requested and all commands are likely to succeed, run multiple tool calls in parallel for optimal performance. The numbered steps below indicate which commands should be batched in parallel.

Git Safety Protocol:
- NEVER update the git config
- NEVER run destructive git commands (push --force, reset --hard, checkout ., restore ., clean -f, branch -D) unless the user explicitly requests these actions. Taking unauthorized destructive actions is unhelpful and can result in lost work, so it's best to ONLY run these commands when given direct instructions
- NEVER skip hooks (--no-verify, --no-gpg-sign, etc) unless the user explicitly requests it
- NEVER run force push to main/master, warn the user if they request it
- CRITICAL: Always create NEW commits rather than amending, unless the user explicitly requests a git amend. When a pre-commit hook fails, the commit did NOT happen — so --amend would modify the PREVIOUS commit, which may result in destroying work or losing previous changes. Instead, after hook failure, fix the issue, re-stage, and create a NEW commit
- When staging files, prefer adding specific files by name rather than using "git add -A" or "git add .", which can accidentally include sensitive files (.env, credentials) or large binaries
- NEVER commit changes unless the user explicitly asks you to. It is VERY IMPORTANT to only commit when explicitly asked, otherwise the user will feel that you are being too proactive

1. Run the following bash commands in parallel, each using the {BASH_TOOL_NAME} tool:
  - Run a git status command to see all untracked files. IMPORTANT: Never use the -uall flag as it can cause memory issues on large repos.
  - Run a git diff command to see both staged and unstaged changes that will be committed.
  - Run a git log command to see recent commit messages, so that you can follow this repository's commit message style.
2. Analyze all staged changes (both previously staged and newly added) and draft a commit message:
  - Summarize the nature of the changes (eg. new feature, enhancement to an existing feature, bug fix, refactoring, test, docs, etc.). Ensure the message accurately reflects the changes and their purpose (i.e. "add" means a wholly new feature, "update" means an enhancement to an existing feature, "fix" means a bug fix, etc.).
  - Do not commit files that likely contain secrets (.env, credentials.json, etc). Warn the user if they specifically request to commit those files
  - Draft a concise (1-2 sentences) commit message that focuses on the "why" rather than the "what"
  - Ensure it accurately reflects the changes and their purpose
3. Run the following commands in parallel:
   - Add relevant untracked files to the staging area.
   - Create the commit with a message.
   - Run git status after the commit completes to verify success.
   Note: git status depends on the commit completing, so run it sequentially after the commit.
4. If the commit fails due to pre-commit hook: fix the issue and create a NEW commit

Important notes:
- NEVER run additional commands to read or explore code, besides git bash commands
- NEVER use the {AGENT_TOOL_NAME} tools
- DO NOT push to the remote repository unless the user explicitly asks you to do so
- IMPORTANT: Never use git commands with the -i flag (like git rebase -i or git add -i) since they require interactive input which is not supported.
- IMPORTANT: Do not use --no-edit with git rebase commands, as the --no-edit flag is not a valid option for git rebase.
- If there are no changes to commit (i.e., no untracked files and no modifications), do not create an empty commit
- In order to ensure good formatting, ALWAYS pass the commit message via a HEREDOC, a la this example:
<example>
git commit -m "$(cat <<'EOF'
   Commit message here.
   EOF
   )"
</example>

# Creating pull requests
Use the gh command via the Bash tool for ALL GitHub-related tasks including working with issues, pull requests, checks, and releases. If given a Github URL use the gh command to get the information needed.

IMPORTANT: When the user asks you to create a pull request, follow these steps carefully:

1. Run the following bash commands in parallel using the {BASH_TOOL_NAME} tool, in order to understand the current state of the branch since it diverged from the main branch:
   - Run a git status command to see all untracked files (never use -uall flag)
   - Run a git diff command to see both staged and unstaged changes that will be committed
   - Check if the current branch tracks a remote branch and is up to date with the remote, so you know if you need to push to the remote
   - Run a git log command and `git diff [base-branch]...HEAD` to understand the full commit history for the current branch (from the time it diverged from the base branch)
2. Analyze all changes that will be included in the pull request, making sure to look at all relevant commits (NOT just the latest commit, but ALL commits that will be included in the pull request!!!), and draft a pull request title and summary:
   - Keep the PR title short (under 70 characters)
   - Use the description/body for details, not the title
3. Run the following commands in parallel:
   - Create new branch if needed
   - Push to remote with -u flag if needed
   - Create PR using gh pr create with the format below. Use a HEREDOC to pass the body to ensure correct formatting.
<example>
gh pr create --title "the pr title" --body "$(cat <<'EOF'
## Summary
<1-3 bullet points>

## Test plan
[Bulleted markdown checklist of TODOs for testing the pull request...]
EOF
)"
</example>

Important:
- Return the PR URL when you're done, so the user can see it

# Other common operations
- View comments on a Github PR: gh api repos/foo/bar/pulls/123/comments"""


class BashTool(Tool):
    name = BASH_TOOL_NAME
    description = _get_bash_prompt()
    risk_level = RiskLevel.HIGH
    is_read_only = False

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": f"Timeout in seconds (default: {DEFAULT_TIMEOUT_S}, max: {MAX_TIMEOUT_S})",
                    "default": DEFAULT_TIMEOUT_S,
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": "Run the command in the background. You will be notified when it completes.",
                    "default": False,
                },
            },
            "required": ["command"],
        }

    def check_permissions(self, arguments: dict[str, Any], config: Any) -> PermissionCheck:
        cmd = arguments.get("command", "")
        if is_dangerous_command(cmd):
            return PermissionCheck(allowed=False, reason=f"Dangerous command blocked: {cmd[:80]}")
        return PermissionCheck(allowed=True)

    async def call(self, arguments: dict[str, Any], context: ToolContext) -> str:
        command = arguments["command"]
        timeout = min(arguments.get("timeout", DEFAULT_TIMEOUT_S), MAX_TIMEOUT_S)
        run_in_background = arguments.get("run_in_background", False)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=context.cwd,
                env={**os.environ},
            )

            if run_in_background:
                # Fire-and-forget: return immediately, notify later
                # (Notification mechanism will be added with Engine integration)
                return f"Command started in background (PID: {proc.pid})"

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

            output_parts = []
            if stdout:
                output_parts.append(stdout.decode(errors="replace"))
            if stderr:
                output_parts.append(f"[stderr]\n{stderr.decode(errors='replace')}")
            if proc.returncode != 0:
                output_parts.append(f"[exit code: {proc.returncode}]")

            return "\n".join(output_parts) if output_parts else "(no output)"

        except asyncio.TimeoutError:
            return f"Command timed out after {timeout}s"
        except Exception as e:
            return f"Error executing command: {e}"
"""
    File tools — 文件操作工具集
    Maps to: src/tools/FileReadTool, FileWriteTool, FileEditTool
"""
