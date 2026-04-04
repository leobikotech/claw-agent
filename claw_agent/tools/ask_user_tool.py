"""
AskUserQuestionTool — 向用户提问工具（人机协作核心）
Maps to: src/tools/AskUserQuestionTool/AskUserQuestionTool.ts

Structured question-answer tool enabling the LLM to pause execution
and ask the user for clarification, decisions, or preferences.
This is the foundation of human-in-the-loop agentic workflows.

结构化问答工具，让 LLM 能暂停执行并向用户征求澄清、决策或偏好。
这是人机协作（Human-in-the-Loop）智能体工作流的基础。

Architecture / 架构:
  LLM decides it needs user input
    → calls ask_user(questions=[...])
    → Engine renders questions in terminal
    → User selects options / types custom answer
    → Engine returns answers to LLM
    → LLM continues with user's choices

Design notes / 设计说明:
  - Original Claude Code uses React (Ink) for TUI rendering
  - We use simple stdin + Rich formatting for the same UX
  - Supports single-select and multi-select questions
  - "Other" custom input is always available
  - Concurrency-safe and non-destructive (read-only tool)
"""
from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any, Optional

from claw_agent.core.tool import Tool, ToolContext, RiskLevel

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# Tool prompt — maps to prompt.ts
# ────────────────────────────────────────────────────────────────

ASK_USER_QUESTION_TOOL_PROMPT = """Use this tool when you need to ask the user questions during execution. This allows you to:
1. Gather user preferences or requirements
2. Clarify ambiguous instructions
3. Get decisions on implementation choices as you work
4. Offer choices to the user about what direction to take

Usage notes:
- Provide 2-4 clear, distinct options for each question
- Users will always be able to select "Other" to provide custom text input
- Use multi_select: true to allow multiple answers to be selected
- If you recommend a specific option, make it the first option and add "(Recommended)" at the end of the label
- Each question should be clear, specific, and end with a question mark
- Do NOT use this tool to ask "Should I proceed?" — just proceed
- Do NOT ask the user questions that you can figure out yourself from context
"""


# ────────────────────────────────────────────────────────────────
# AskUserQuestionTool
# ────────────────────────────────────────────────────────────────

class AskUserQuestionTool(Tool):
    """Ask User Question Tool — 向用户提问工具
    Maps to: src/tools/AskUserQuestionTool/AskUserQuestionTool.ts

    Enables the LLM to present structured multiple-choice questions to the
    user and receive their selections. This is essential for human-in-the-loop
    workflows where the agent needs clarification or user decisions.

    让 LLM 能向用户展示结构化的多选题并接收选择。
    这是人机协作工作流中不可或缺的工具。

    Input schema (from LLM):
      questions: [{
          question: str,        # The question text
          options: [{
              label: str,       # Short display label (1-5 words)
              description: str  # Explanation of the option
          }],
          multi_select: bool    # Allow multiple selections?
      }]

    Output (to LLM):
      Formatted text of "question" → "answer" pairs
    """

    name = "ask_user"
    description = (
        "Ask the user multiple choice questions to gather information, "
        "clarify ambiguity, understand preferences, make decisions or "
        "offer them choices. Users can always provide custom text input."
    )
    risk_level = RiskLevel.LOW
    is_read_only = True

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": (
                                    "The complete question to ask. Should be clear, "
                                    "specific, and end with a question mark."
                                ),
                            },
                            "options": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": {
                                            "type": "string",
                                            "description": "Short display text (1-5 words)",
                                        },
                                        "description": {
                                            "type": "string",
                                            "description": "Explanation of what this option means",
                                        },
                                    },
                                    "required": ["label", "description"],
                                },
                                "minItems": 2,
                                "maxItems": 4,
                                "description": "Available choices (2-4 options). Do not include 'Other', it is added automatically.",
                            },
                            "multi_select": {
                                "type": "boolean",
                                "default": False,
                                "description": "Allow multiple selections? Use when choices are not mutually exclusive.",
                            },
                        },
                        "required": ["question", "options"],
                    },
                    "minItems": 1,
                    "maxItems": 4,
                    "description": "Questions to ask the user (1-4 questions)",
                },
            },
            "required": ["questions"],
        }

    async def call(self, arguments: dict[str, Any], context: ToolContext) -> str:
        questions = arguments.get("questions", [])

        if not questions:
            return "Error: No questions provided."

        # Validate questions
        for q in questions:
            if not q.get("question"):
                return "Error: Each question must have a 'question' text."
            if not q.get("options") or len(q["options"]) < 2:
                return "Error: Each question must have at least 2 options."

        answers: dict[str, str] = {}

        for i, q in enumerate(questions, 1):
            question_text = q["question"]
            options = q["options"]
            multi_select = q.get("multi_select", False)

            answer = await _ask_question_interactive(
                question_text=question_text,
                options=options,
                multi_select=multi_select,
                question_number=i,
                total_questions=len(questions),
            )

            answers[question_text] = answer

        # Format result for LLM
        result_parts = []
        for question_text, answer in answers.items():
            result_parts.append(f'"{question_text}" → "{answer}"')

        answers_text = "; ".join(result_parts)
        return (
            f"User has answered your questions: {answers_text}. "
            f"You can now continue with the user's answers in mind."
        )

    def get_prompt(self) -> str:
        """Return the tool prompt / 返回工具提示词"""
        return ASK_USER_QUESTION_TOOL_PROMPT


# ────────────────────────────────────────────────────────────────
# Interactive question rendering — 交互式问题渲染
# ────────────────────────────────────────────────────────────────

async def _ask_question_interactive(
    question_text: str,
    options: list[dict],
    multi_select: bool,
    question_number: int,
    total_questions: int,
) -> str:
    """Render a question in the terminal and collect user input.
    在终端渲染问题并收集用户输入。

    Uses Rich for styled output + raw stdin for input collection.
    Uses asyncio.to_thread to avoid blocking the event loop.
    """

    def _blocking_ask() -> str:
        """Synchronous blocking input — runs in thread to not block asyncio."""
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.text import Text

            c = Console()
            c.print()

            # Question header
            header = f"❓ Question {question_number}/{total_questions}"
            if multi_select:
                header += "  [dim](multi-select: enter numbers separated by commas)[/dim]"

            c.print(f"  [bold cyan]{header}[/bold cyan]")
            c.print(f"  [bold white]{question_text}[/bold white]")
            c.print()

            # Render options
            for j, opt in enumerate(options, 1):
                label = opt.get("label", f"Option {j}")
                desc = opt.get("description", "")
                c.print(f"    [bold yellow]{j}[/bold yellow]. [bold]{label}[/bold]")
                if desc:
                    c.print(f"       [dim]{desc}[/dim]")

            # Always add "Other" option
            other_num = len(options) + 1
            c.print(f"    [bold yellow]{other_num}[/bold yellow]. [bold]Other[/bold]")
            c.print(f"       [dim]Provide your own custom answer[/dim]")
            c.print()

            # Collect input
            while True:
                try:
                    if multi_select:
                        raw = input(f"  Select option(s) [1-{other_num}, comma-separated]: ").strip()
                    else:
                        raw = input(f"  Select option [1-{other_num}]: ").strip()
                except (EOFError, KeyboardInterrupt):
                    return "No answer (user skipped)"

                if not raw:
                    continue

                if multi_select:
                    # Parse comma-separated numbers
                    try:
                        selections = [int(x.strip()) for x in raw.split(",")]
                    except ValueError:
                        c.print("  [red]Please enter numbers separated by commas.[/red]")
                        continue

                    # Validate all selections
                    invalid = [s for s in selections if s < 1 or s > other_num]
                    if invalid:
                        c.print(f"  [red]Invalid option(s): {invalid}. Choose 1-{other_num}.[/red]")
                        continue

                    # If "Other" is among selections, ask for custom text
                    if other_num in selections:
                        custom = input("  Your custom answer: ").strip()
                        selections = [s for s in selections if s != other_num]
                        labels = [options[s - 1]["label"] for s in selections if 1 <= s <= len(options)]
                        if custom:
                            labels.append(custom)
                        return ", ".join(labels) if labels else custom or "No answer"
                    else:
                        labels = [options[s - 1]["label"] for s in selections if 1 <= s <= len(options)]
                        return ", ".join(labels)

                else:
                    # Single select
                    try:
                        choice = int(raw)
                    except ValueError:
                        c.print(f"  [red]Please enter a number 1-{other_num}.[/red]")
                        continue

                    if choice < 1 or choice > other_num:
                        c.print(f"  [red]Please choose 1-{other_num}.[/red]")
                        continue

                    if choice == other_num:
                        custom = input("  Your custom answer: ").strip()
                        return custom or "No answer"
                    else:
                        return options[choice - 1]["label"]

        except Exception as e:
            logger.error(f"AskUser interactive error: {e}")
            return f"Error collecting user input: {e}"

    # Run blocking input in a thread to not block the asyncio event loop
    return await asyncio.to_thread(_blocking_ask)
