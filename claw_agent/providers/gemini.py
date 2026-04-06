"""
Google Gemini Provider — Gemini API

Uses the google-genai SDK with manual tool handling.
使用 google-genai SDK，手动处理工具调用。
"""
from __future__ import annotations

import json
import logging
from typing import Any

from claw_agent.providers.base import LLMProvider, LLMResponse, LLMToolCall

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Provider for Google Gemini / Gemini API Provider
    Uses the google-genai SDK with manual tool handling.
    使用 google-genai SDK，手动处理工具调用。
    """

    name = "gemini"

    def __init__(self, api_key: str, **kwargs):
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._genai = genai

    async def chat(self, messages: list[dict], tools: list[dict] | None = None, **kwargs) -> LLMResponse:
        from google.genai import types
        import asyncio

        model = kwargs.get("model", "gemini-2.5-flash")

        # Convert OpenAI messages → Gemini contents
        contents = []
        system_instruction = None
        for m in messages:
            if m["role"] == "system":
                system_instruction = m["content"]
            elif m["role"] == "user":
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=m["content"])]))
            elif m["role"] == "assistant":
                parts = []
                if m.get("content"):
                    parts.append(types.Part.from_text(text=m["content"]))
                if m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        fn = tc["function"] if isinstance(tc.get("function"), dict) else tc
                        args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                        parts.append(types.Part.from_function_call(name=fn["name"], args=args))
                contents.append(types.Content(role="model", parts=parts))
            elif m["role"] == "tool":
                # Look up the function name from the tool_call_id by scanning
                # preceding assistant messages for matching tool calls
                fn_name = "tool"  # fallback
                tool_call_id = m.get("tool_call_id", "")
                for prev in reversed(messages[:messages.index(m)]):
                    if prev.get("role") == "assistant" and prev.get("tool_calls"):
                        for tc in prev["tool_calls"]:
                            if tc.get("id") == tool_call_id:
                                fn = tc.get("function", tc)
                                fn_name = fn.get("name", "tool") if isinstance(fn, dict) else "tool"
                                break
                        break
                parts = [types.Part.from_function_response(
                    name=fn_name,
                    response={"result": m["content"]},
                )]
                contents.append(types.Content(role="user", parts=parts))

        # Convert OpenAI tools → Gemini tool declarations
        gemini_tools = None
        if tools:
            declarations = []
            for t in tools:
                fn = t["function"]
                declarations.append(types.FunctionDeclaration(
                    name=fn["name"],
                    description=fn.get("description", ""),
                    parameters=fn.get("parameters"),
                ))
            gemini_tools = [types.Tool(function_declarations=declarations)]

        config = types.GenerateContentConfig(
            temperature=kwargs.get("temperature", 0.0),
            max_output_tokens=kwargs.get("max_tokens", 4096),
            tools=gemini_tools,  # type: ignore[arg-type]
        )
        if system_instruction:
            config.system_instruction = system_instruction

        # Gemini SDK is sync by default — run in executor
        response = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self._client.models.generate_content(
                model=model, contents=contents, config=config,
            ),
        )

        # Normalize
        content_text = ""
        tool_calls = []
        if response.candidates:
            parts = getattr(response.candidates[0].content, "parts", [])
            if parts:
                for part in parts:
                    if part.text:
                        content_text += part.text
                    elif part.function_call:
                        fc = part.function_call
                        tool_calls.append(LLMToolCall(
                            id=f"gemini_{fc.name}_{id(fc)}",
                            name=fc.name or "unknown",
                            arguments=dict(fc.args) if fc.args else {},
                        ))

        return LLMResponse(
            content=content_text or None,
            tool_calls=tool_calls,
            finish_reason="stop",
            usage={},
        )
