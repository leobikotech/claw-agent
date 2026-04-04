"""
WebFetchTool — HTTP 请求工具
Maps to: src/tools/WebFetchTool/WebFetchTool.ts

Fetch content from URLs via HTTP.
通过 HTTP 获取 URL 内容。
"""
from __future__ import annotations
from typing import Any

import httpx

from claw_agent.core.tool import Tool, ToolContext, RiskLevel


class WebFetchTool(Tool):
    name = "web_fetch"
    description = "Fetch content from a URL. Returns the response body as text."
    risk_level = RiskLevel.LOW
    is_read_only = True

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch"},
                "method": {"type": "string", "enum": ["GET", "POST"], "default": "GET"},
                "headers": {"type": "object", "description": "Optional HTTP headers"},
            },
            "required": ["url"],
        }

    async def call(self, arguments: dict[str, Any], context: ToolContext) -> str:
        url = arguments["url"]
        method = arguments.get("method", "GET")
        headers = arguments.get("headers", {})

        try:
            default_headers = {"User-Agent": "CrawAgent/0.2 (httpx)"}
            default_headers.update(headers)
            async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
                resp = await client.request(method, url, headers=default_headers)
                # Truncate very large responses
                text = resp.text[:50_000]
                if len(resp.text) > 50_000:
                    text += f"\n... (truncated, {len(resp.text)} total chars)"
                return f"[{resp.status_code}]\n{text}"
        except Exception as e:
            return f"Error fetching URL: {e}"
