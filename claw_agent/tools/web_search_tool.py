"""
WebSearchTool — 网络搜索工具（现代 Agent 核心能力）

Provider-agnostic web search with pluggable backends.
Some agent systems use provider-native web search APIs,
which are provider-specific. This implementation provides the same
capability via standard search APIs, making it work with ANY LLM provider.

Supported backends (auto-detected from env vars):
  1. Exa      — AI-native neural search, returns extracted content (EXA_API_KEY)
  2. Tavily   — AI-optimized search, returns extracted content (TAVILY_API_KEY)
  3. Brave    — Privacy-focused, generous free tier (BRAVE_API_KEY)
  4. SerpAPI  — Google results via API (SERPAPI_API_KEY)
  5. DuckDuckGo — Zero config, no API key needed (fallback)

Architecture / 架构:
  LLM → WebSearchTool.call(query) → SearchBackend → structured results
  LLM 调用工具 → 搜索后端返回结构化结果 → 格式化为带来源链接的文本

Design notes / 设计说明:
  - Results include extracted page content, not just links
  - Sources are formatted as markdown hyperlinks for LLM citation
  - Domain filtering (allow/block) supported across all backends
  - Automatic year injection for time-sensitive queries
"""
from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import httpx

from claw_agent.core.tool import Tool, ToolContext, RiskLevel

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# Data structures — 搜索结果数据结构
# ────────────────────────────────────────────────────────────────

@dataclass
class SearchHit:
    """A single search result / 单条搜索结果"""
    title: str
    url: str
    snippet: str = ""          # Short text snippet / 短文本摘要
    content: str = ""          # Full extracted content (if available) / 完整提取内容
    score: float = 0.0         # Relevance score (0-1, backend dependent)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"title": self.title, "url": self.url, "snippet": self.snippet}
        if self.content:
            d["content"] = self.content
        if self.score:
            d["score"] = round(self.score, 3)
        return d


@dataclass
class SearchResult:
    """Aggregated search response / 聚合搜索响应"""
    query: str
    hits: list[SearchHit] = field(default_factory=list)
    answer: str = ""           # AI-generated answer (Tavily) / AI 生成的答案
    duration_seconds: float = 0.0

    def format_for_llm(self, max_content_chars: int = 4000) -> str:
        """Format results for LLM consumption / 格式化结果供 LLM 使用

        Follows the standard agent pattern:
        - Structured results with titles and URLs
        - Content snippets for context
        - Sources section at the end with markdown hyperlinks
        - Reminder to cite sources
        """
        parts = [f'Web search results for: "{self.query}"\n']

        if self.answer:
            parts.append(f"AI Summary:\n{self.answer}\n")

        if not self.hits:
            parts.append("No results found.")
            return "\n".join(parts)

        parts.append(f"Found {len(self.hits)} results:\n")

        for i, hit in enumerate(self.hits, 1):
            parts.append(f"--- Result {i} ---")
            parts.append(f"Title: {hit.title}")
            parts.append(f"URL: {hit.url}")

            # Prefer full content, fall back to snippet
            text = hit.content or hit.snippet
            if text:
                # Truncate very long content per result
                if len(text) > max_content_chars:
                    text = text[:max_content_chars] + "... (truncated)"
                parts.append(f"Content:\n{text}")
            parts.append("")

        # Sources section — critical for citation
        parts.append("Sources:")
        for hit in self.hits:
            parts.append(f"- [{hit.title}]({hit.url})")

        parts.append(
            "\nREMINDER: You MUST include the sources above in your response "
            "to the user using markdown hyperlinks."
        )

        return "\n".join(parts)


# ────────────────────────────────────────────────────────────────
# Search backend abstraction — 搜索后端抽象
# ────────────────────────────────────────────────────────────────

class SearchBackend(ABC):
    """Base class for search backends / 搜索后端基类"""

    name: str = "base"

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 5,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> list[SearchHit]:
        """Execute a search query / 执行搜索查询"""
        ...


# ────────────────────────────────────────────────────────────────
# Backend: Exa — AI-native neural search with full content extraction
# ────────────────────────────────────────────────────────────────

class ExaBackend(SearchBackend):
    """Exa Search API — neural search built for agents / 专为智能体设计的神经搜索

    Embedding-based search that understands intent rather than keywords.
    Returns extracted page text, highlights, and AI-generated summaries
    in a single call — ideal for RAG and agent retrieval pipelines.
    基于向量嵌入的搜索，可一次性返回正文、高亮片段与 AI 摘要。

    Docs: https://exa.ai/docs/reference/search
    """

    name = "exa"

    # Valid search types per current Exa API reference.
    # `auto` intelligently blends neural + keyword; `fast` is low-latency;
    # `neural` is pure embeddings; `deep`/`deep-lite` synthesize richer output.
    _VALID_TYPES = {
        "auto", "neural", "fast",
        "deep", "deep-lite", "deep-reasoning", "instant",
    }

    def __init__(
        self,
        api_key: str,
        search_type: str = "auto",
        category: str | None = None,
        highlight_chars: int = 500,
        text_chars: int = 2000,
    ):
        self.api_key = api_key
        self.search_type = search_type if search_type in self._VALID_TYPES else "auto"
        self.category = category
        self.highlight_chars = highlight_chars
        self.text_chars = text_chars
        self.endpoint = "https://api.exa.ai/search"

    async def search(
        self,
        query: str,
        max_results: int = 5,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> list[SearchHit]:
        payload: dict[str, Any] = {
            "query": query,
            "type": self.search_type,
            "numResults": max_results,
            # Request highlights + compact text together — Exa allows both
            # in one call so we can populate snippet and full content fields.
            "contents": {
                "highlights": {"maxCharacters": self.highlight_chars},
                "text": {
                    "maxCharacters": self.text_chars,
                    "verbosity": "compact",
                },
            },
        }
        if self.category:
            payload["category"] = self.category
        if allowed_domains:
            payload["includeDomains"] = allowed_domains
        if blocked_domains:
            payload["excludeDomains"] = blocked_domains

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            # Integration attribution header — identifies traffic from this client.
            "x-exa-integration": "claw-agent",
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(self.endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        hits = []
        for r in data.get("results", []):
            hits.append(self._parse_result(r))
        return hits

    @staticmethod
    def _parse_result(r: dict[str, Any]) -> SearchHit:
        """Parse a single Exa result with graceful content fallbacks.

        Exa may return any combination of `highlights`, `text`, `summary`.
        We cascade: highlights → summary → first chars of text for snippet,
        and prefer full text for the content field. This matches how the
        rest of this file handles partial-content responses.
        """
        highlights = r.get("highlights") or []
        summary = r.get("summary") or ""
        text = r.get("text") or ""

        if highlights:
            snippet = " … ".join(h for h in highlights if h)[:1000]
        elif summary:
            snippet = summary[:1000]
        elif text:
            snippet = text[:500]
        else:
            snippet = ""

        content = text or summary or " ".join(highlights)

        return SearchHit(
            title=r.get("title") or "",
            url=r.get("url") or "",
            snippet=snippet,
            content=content,
            score=float(r.get("score") or 0.0),
        )


# ────────────────────────────────────────────────────────────────
# Backend: Tavily — AI-optimized search (recommended)
# ────────────────────────────────────────────────────────────────

class TavilyBackend(SearchBackend):
    """Tavily Search API — designed for AI agents / 专为 AI 智能体设计

    Best for agents: returns extracted page content, not just links.
    Free tier: 1000 searches/month.
    对 Agent 最友好：返回提取的页面内容而非仅链接。
    """

    name = "tavily"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://api.tavily.com/search"

    async def search(
        self,
        query: str,
        max_results: int = 5,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> list[SearchHit]:
        payload: dict[str, Any] = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "advanced",       # Get full page content
            "include_answer": True,           # AI-generated answer
            "include_raw_content": False,      # Save bandwidth
        }
        if allowed_domains:
            payload["include_domains"] = allowed_domains
        if blocked_domains:
            payload["exclude_domains"] = blocked_domains

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(self.endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()

        hits = []
        for r in data.get("results", []):
            hits.append(SearchHit(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", "")[:500],
                content=r.get("content", ""),
                score=r.get("score", 0.0),
            ))

        return hits


# ────────────────────────────────────────────────────────────────
# Backend: Brave Search — privacy-focused, free tier
# ────────────────────────────────────────────────────────────────

class BraveBackend(SearchBackend):
    """Brave Search API — privacy-first

    Free tier: 2000 queries/month.
    免费层级：2000 次/月。
    """

    name = "brave"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://api.search.brave.com/res/v1/web/search"

    async def search(
        self,
        query: str,
        max_results: int = 5,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> list[SearchHit]:
        # Apply domain filtering via query syntax
        effective_query = query
        if allowed_domains:
            site_filter = " OR ".join(f"site:{d}" for d in allowed_domains)
            effective_query = f"({query}) ({site_filter})"
        if blocked_domains:
            block_filter = " ".join(f"-site:{d}" for d in blocked_domains)
            effective_query = f"{query} {block_filter}"

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }
        params = {"q": effective_query, "count": max_results}

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(self.endpoint, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        hits = []
        for r in data.get("web", {}).get("results", []):
            hits.append(SearchHit(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("description", ""),
            ))

        return hits


# ────────────────────────────────────────────────────────────────
# Backend: SerpAPI — Google Search results via API
# ────────────────────────────────────────────────────────────────

class SerpAPIBackend(SearchBackend):
    """SerpAPI — Google Search results / Google 搜索结果

    Uses SerpAPI to access Google Search.
    """

    name = "serpapi"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://serpapi.com/search"

    async def search(
        self,
        query: str,
        max_results: int = 5,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> list[SearchHit]:
        effective_query = query
        if allowed_domains:
            site_filter = " OR ".join(f"site:{d}" for d in allowed_domains)
            effective_query = f"({query}) ({site_filter})"
        if blocked_domains:
            block_filter = " ".join(f"-site:{d}" for d in blocked_domains)
            effective_query = f"{query} {block_filter}"

        params = {
            "q": effective_query,
            "api_key": self.api_key,
            "engine": "google",
            "num": max_results,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(self.endpoint, params=params)
            resp.raise_for_status()
            data = resp.json()

        hits = []
        for r in data.get("organic_results", []):
            hits.append(SearchHit(
                title=r.get("title", ""),
                url=r.get("link", ""),
                snippet=r.get("snippet", ""),
            ))

        return hits


# ────────────────────────────────────────────────────────────────
# Backend: DuckDuckGo — zero config fallback (no API key)
# ────────────────────────────────────────────────────────────────

class DuckDuckGoBackend(SearchBackend):
    """DuckDuckGo HTML scraper — no API key needed / 无需 API key

    Uses DuckDuckGo's lite HTML interface as a zero-config fallback.
    Parses search results from the HTML response.
    作为零配置后备方案，解析 DuckDuckGo 的 HTML 搜索结果。

    Note: Rate limited. For production use, prefer Tavily or Brave.
    注意：有频率限制。生产环境推荐使用 Tavily 或 Brave。
    """

    name = "duckduckgo"

    async def search(
        self,
        query: str,
        max_results: int = 5,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> list[SearchHit]:
        import re
        import html as html_mod
        from urllib.parse import urlparse, parse_qs, unquote

        effective_query = query
        if allowed_domains:
            site_filter = " OR ".join(f"site:{d}" for d in allowed_domains)
            effective_query = f"{query} ({site_filter})"
        if blocked_domains:
            block_filter = " ".join(f"-site:{d}" for d in blocked_domains)
            effective_query = f"{query} {block_filter}"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        }

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(
                "https://lite.duckduckgo.com/lite/",
                params={"q": effective_query},
                headers=headers,
            )
            resp.raise_for_status()
            raw_html = resp.text

        # Parse results from DuckDuckGo Lite HTML
        hits = []

        # Pattern: each result has a link and snippet in the table
        # Links: <a rel="nofollow" href="URL" class='result-link'>TITLE</a>
        link_pattern = re.compile(
            r'<a\s+rel="nofollow"\s+href="([^"]+)"\s+class=\'result-link\'>(.+?)</a>',
            re.DOTALL,
        )
        # Snippets: <td class="result-snippet">TEXT</td>
        snippet_pattern = re.compile(
            r'<td\s+class="result-snippet">\s*(.*?)\s*</td>',
            re.DOTALL,
        )

        links = link_pattern.findall(raw_html)
        snippets = snippet_pattern.findall(raw_html)

        for i, (raw_url, title) in enumerate(links[:max_results]):
            # Clean HTML tags and decode HTML entities
            clean_title = html_mod.unescape(re.sub(r'<[^>]+>', '', title).strip())
            clean_snippet = ""
            if i < len(snippets):
                clean_snippet = html_mod.unescape(
                    re.sub(r'<[^>]+>', '', snippets[i]).strip()
                )

            # Extract real URL from DuckDuckGo redirect wrapper
            # DDG wraps URLs like: //duckduckgo.com/l/?uddg=https%3A%2F%2F...&rut=...
            url = html_mod.unescape(raw_url)
            parsed = urlparse(url)
            if "duckduckgo.com" in parsed.netloc or (
                not parsed.scheme and "duckduckgo.com" in parsed.path
            ):
                qs = parse_qs(parsed.query)
                if "uddg" in qs:
                    url = unquote(qs["uddg"][0])
                elif "/y.js" in url:
                    # Skip ad/tracking URLs (duckduckgo.com/y.js)
                    continue

            # Ensure URL has scheme
            if url.startswith("//"):
                url = "https:" + url

            # Final guard: skip any remaining DDG internal URLs
            if "duckduckgo.com" in url:
                continue

            if url and clean_title:
                hits.append(SearchHit(
                    title=clean_title,
                    url=url,
                    snippet=clean_snippet,
                ))

        return hits


# ────────────────────────────────────────────────────────────────
# Backend auto-detection — 后端自动检测
# ────────────────────────────────────────────────────────────────

def auto_detect_backend() -> SearchBackend:
    """Auto-detect the best available search backend from env vars.
    从环境变量自动检测最佳搜索后端。

    Priority: Exa > Tavily > Brave > SerpAPI > DuckDuckGo (free fallback)
    """
    if key := os.environ.get("EXA_API_KEY"):
        logger.info("WebSearch: using Exa backend (neural search)")
        return ExaBackend(api_key=key)

    if key := os.environ.get("TAVILY_API_KEY"):
        logger.info("WebSearch: using Tavily backend (AI-optimized)")
        return TavilyBackend(api_key=key)

    if key := os.environ.get("BRAVE_API_KEY"):
        logger.info("WebSearch: using Brave Search backend")
        return BraveBackend(api_key=key)

    if key := os.environ.get("SERPAPI_API_KEY"):
        logger.info("WebSearch: using SerpAPI (Google) backend")
        return SerpAPIBackend(api_key=key)

    logger.info("WebSearch: using DuckDuckGo fallback (no API key needed)")
    return DuckDuckGoBackend()


# ────────────────────────────────────────────────────────────────
# WebSearchTool — the tool that LLMs call
# ────────────────────────────────────────────────────────────────

def _get_web_search_prompt() -> str:
    current_month_year = datetime.now().strftime("%B %Y")
    return f"""WebSearch — search the web for current information.
- Provides up-to-date information for current events and recent data
- Returns search results with titles, URLs, and content snippets
- Use this tool for accessing information beyond your knowledge cutoff
- Supports domain filtering to include or block specific websites

CRITICAL REQUIREMENT — You MUST follow this:
  - After answering the user's question, you MUST include a "Sources:" section
  - In Sources, list all relevant URLs as markdown hyperlinks: [Title](URL)
  - This is MANDATORY — never skip including sources in your response
  - Example format:

    [Your answer here]

    Sources:
    - [Source Title 1](https://example.com/1)
    - [Source Title 2](https://example.com/2)

IMPORTANT — Use the correct year in search queries:
  - The current month is {current_month_year}. Use this year when searching
    for recent information, documentation, or current events.
"""


class WebSearchTool(Tool):
    """Web Search Tool — 网络搜索工具（现代 Agent 核心能力）

    Provider-agnostic web search with pluggable backends.
    Some agent systems use provider-native web search APIs.
    This implementation uses standard search APIs, working with ANY provider.

    部分智能体系统使用供应商原生的 web_search API。
    本实现使用标准搜索 API，兼容任何 LLM Provider。

    Backend auto-detection (via env vars):
      EXA_API_KEY     → Exa (AI-native neural search)
      TAVILY_API_KEY  → Tavily (AI-optimized)
      BRAVE_API_KEY   → Brave Search
      SERPAPI_API_KEY  → SerpAPI (Google results)
      (none)          → DuckDuckGo (free fallback)
    """

    name = "web_search"
    description = (
        "Search the web for current information. Returns search results with "
        "titles, URLs, and content snippets. Use this for accessing information "
        "beyond your knowledge cutoff, such as current events, recent documentation, "
        "latest versions, or any time-sensitive data."
    )
    risk_level = RiskLevel.LOW
    is_read_only = True

    def __init__(self, backend: SearchBackend | None = None):
        """Initialize with optional explicit backend / 可选指定搜索后端

        Args:
            backend: Explicit search backend. If None, auto-detects from env vars.
        """
        self._backend = backend

    @property
    def backend(self) -> SearchBackend:
        """Lazy init backend (so env vars can be set after import)."""
        if self._backend is None:
            self._backend = auto_detect_backend()
        return self._backend

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to use. Be specific and include the current year for time-sensitive queries.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5, max: 10)",
                    "default": 5,
                },
                "allowed_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only include results from these domains (e.g. ['github.com', 'stackoverflow.com'])",
                },
                "blocked_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exclude results from these domains",
                },
            },
            "required": ["query"],
        }

    async def call(self, arguments: dict[str, Any], context: ToolContext) -> str:
        query = arguments["query"]
        max_results = min(arguments.get("max_results", 5), 10)
        allowed_domains = arguments.get("allowed_domains")
        blocked_domains = arguments.get("blocked_domains")

        # Validate: cannot specify both allow and block
        if allowed_domains and blocked_domains:
            return "Error: Cannot specify both allowed_domains and blocked_domains in the same request."

        if not query or len(query.strip()) < 2:
            return "Error: Search query must be at least 2 characters."

        start_time = time.time()

        try:
            hits = await self.backend.search(
                query=query,
                max_results=max_results,
                allowed_domains=allowed_domains,
                blocked_domains=blocked_domains,
            )

            duration = time.time() - start_time

            result = SearchResult(
                query=query,
                hits=hits,
                duration_seconds=round(duration, 2),
            )

            # If using Tavily, try to get the AI answer
            if isinstance(self.backend, TavilyBackend):
                result.answer = getattr(self.backend, '_last_answer', '')

            formatted = result.format_for_llm()

            logger.info(
                f"WebSearch: '{query}' → {len(hits)} results in {duration:.1f}s "
                f"(backend={self.backend.name})"
            )

            return formatted

        except httpx.HTTPStatusError as e:
            logger.error(f"WebSearch HTTP error: {e.response.status_code} for '{query}'")
            return (
                f"Web search failed (HTTP {e.response.status_code}). "
                f"Please check your {self.backend.name} API key and try again."
            )
        except httpx.TimeoutException:
            logger.error(f"WebSearch timeout for '{query}'")
            return "Web search timed out. Please try again with a simpler query."
        except Exception as e:
            logger.error(f"WebSearch error: {e}")
            return f"Web search failed: {str(e)}"

    def get_prompt(self) -> str:
        """Return the web search tool prompt / 返回搜索工具提示词"""
        return _get_web_search_prompt()
