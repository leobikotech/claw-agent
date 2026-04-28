"""Tests for the Exa search backend.

Covers response parsing, content fallbacks, auto-detection, and request
payload shape. All HTTP calls are mocked — no live API calls are made.
"""
from __future__ import annotations

import json
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from claw_agent.tools.web_search_tool import (
    ExaBackend,
    SearchHit,
    auto_detect_backend,
    TavilyBackend,
    DuckDuckGoBackend,
)


# A realistic response fragment mirroring the Exa /search JSON shape.
SAMPLE_RESPONSE = {
    "results": [
        {
            "title": "Attention Is All You Need",
            "url": "https://arxiv.org/abs/1706.03762",
            "publishedDate": "2017-06-12T00:00:00.000Z",
            "author": "Vaswani et al.",
            "score": 0.87,
            "highlights": [
                "We propose a new simple network architecture, the Transformer",
                "based solely on attention mechanisms",
            ],
            "text": "The dominant sequence transduction models are based on recurrent "
                    "or convolutional neural networks. We propose a new architecture...",
        },
        {
            "title": "A Result With Only Text",
            "url": "https://example.com/text-only",
            "score": 0.5,
            "text": "Some body text from the page without highlights or summary.",
        },
        {
            "title": "A Result With Only Summary",
            "url": "https://example.com/summary-only",
            "score": 0.3,
            "summary": "A concise AI-generated overview of the page.",
        },
        {
            "title": "Sparse Result",
            "url": "https://example.com/sparse",
        },
    ],
    "costDollars": {"total": 0.005},
}


def _mock_httpx_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload)
    return resp


@pytest.mark.asyncio
async def test_parse_full_response():
    """End-to-end parse: mocked POST returns the fixture, backend yields SearchHits."""
    backend = ExaBackend(api_key="test-key")

    resp = _mock_httpx_response(SAMPLE_RESPONSE)
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "claw_agent.tools.web_search_tool.httpx.AsyncClient",
        return_value=mock_client,
    ):
        hits = await backend.search(query="transformer architecture", max_results=4)

    assert len(hits) == 4
    assert all(isinstance(h, SearchHit) for h in hits)

    # Verify request payload shape was built correctly.
    called_with = mock_client.post.call_args
    body = called_with.kwargs["json"]
    assert body["query"] == "transformer architecture"
    assert body["type"] == "auto"
    assert body["numResults"] == 4
    # Exa accepts text + highlights simultaneously — verify both are present.
    assert "highlights" in body["contents"]
    assert "text" in body["contents"]
    assert body["contents"]["text"]["verbosity"] == "compact"

    # Integration attribution header must be present.
    headers = called_with.kwargs["headers"]
    assert headers["x-exa-integration"] == "claw-agent"
    assert headers["x-api-key"] == "test-key"


@pytest.mark.asyncio
async def test_content_fallback_highlights_preferred():
    """When highlights are present, snippet joins them; content prefers text."""
    backend = ExaBackend(api_key="k")
    resp = _mock_httpx_response(SAMPLE_RESPONSE)
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "claw_agent.tools.web_search_tool.httpx.AsyncClient",
        return_value=mock_client,
    ):
        hits = await backend.search(query="q")

    # Result 0: has highlights + text → snippet from highlights, content from text.
    assert "Transformer" in hits[0].snippet
    assert "attention mechanisms" in hits[0].snippet
    assert "dominant sequence transduction" in hits[0].content
    assert hits[0].score == pytest.approx(0.87)


@pytest.mark.asyncio
async def test_content_fallback_text_only():
    """Missing highlights/summary → snippet falls back to text prefix."""
    backend = ExaBackend(api_key="k")
    resp = _mock_httpx_response(SAMPLE_RESPONSE)
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "claw_agent.tools.web_search_tool.httpx.AsyncClient",
        return_value=mock_client,
    ):
        hits = await backend.search(query="q")

    text_only = hits[1]
    assert "Some body text" in text_only.snippet
    assert text_only.content == "Some body text from the page without highlights or summary."


@pytest.mark.asyncio
async def test_content_fallback_summary_only():
    """Summary-only results use summary for both snippet and content."""
    backend = ExaBackend(api_key="k")
    resp = _mock_httpx_response(SAMPLE_RESPONSE)
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "claw_agent.tools.web_search_tool.httpx.AsyncClient",
        return_value=mock_client,
    ):
        hits = await backend.search(query="q")

    summary_only = hits[2]
    assert summary_only.snippet == "A concise AI-generated overview of the page."
    assert summary_only.content == "A concise AI-generated overview of the page."


@pytest.mark.asyncio
async def test_sparse_result_does_not_crash():
    """A result with no content fields still produces a valid SearchHit."""
    backend = ExaBackend(api_key="k")
    resp = _mock_httpx_response(SAMPLE_RESPONSE)
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "claw_agent.tools.web_search_tool.httpx.AsyncClient",
        return_value=mock_client,
    ):
        hits = await backend.search(query="q")

    sparse = hits[3]
    assert sparse.title == "Sparse Result"
    assert sparse.url == "https://example.com/sparse"
    assert sparse.snippet == ""
    assert sparse.content == ""
    assert sparse.score == 0.0


@pytest.mark.asyncio
async def test_domain_filters_mapped_to_exa_fields():
    """allowed_domains/blocked_domains map to includeDomains/excludeDomains."""
    backend = ExaBackend(api_key="k")
    resp = _mock_httpx_response({"results": []})
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "claw_agent.tools.web_search_tool.httpx.AsyncClient",
        return_value=mock_client,
    ):
        await backend.search(
            query="q",
            allowed_domains=["arxiv.org", "nature.com"],
        )
        await backend.search(
            query="q",
            blocked_domains=["example.com"],
        )

    first = mock_client.post.call_args_list[0].kwargs["json"]
    second = mock_client.post.call_args_list[1].kwargs["json"]
    assert first["includeDomains"] == ["arxiv.org", "nature.com"]
    assert "excludeDomains" not in first
    assert second["excludeDomains"] == ["example.com"]
    assert "includeDomains" not in second


def test_invalid_search_type_falls_back_to_auto():
    """Unknown search types are silently coerced to `auto` per API safety."""
    backend = ExaBackend(api_key="k", search_type="keyword")  # removed from API
    assert backend.search_type == "auto"

    backend_ok = ExaBackend(api_key="k", search_type="neural")
    assert backend_ok.search_type == "neural"


def test_category_included_when_set(monkeypatch):
    """category is only sent when explicitly configured."""
    backend_cat = ExaBackend(api_key="k", category="research paper")
    assert backend_cat.category == "research paper"

    backend_plain = ExaBackend(api_key="k")
    assert backend_plain.category is None


def test_auto_detect_picks_exa_when_key_set(monkeypatch):
    """Auto-detection prioritises Exa when EXA_API_KEY is present."""
    monkeypatch.setenv("EXA_API_KEY", "exa-test-key")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-key")  # lower priority

    backend = auto_detect_backend()
    assert isinstance(backend, ExaBackend)
    assert backend.api_key == "exa-test-key"


def test_auto_detect_skips_exa_when_unset(monkeypatch):
    """Without EXA_API_KEY, auto-detection falls through to the next backend."""
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-key")

    backend = auto_detect_backend()
    assert isinstance(backend, TavilyBackend)


def test_auto_detect_defaults_to_duckduckgo(monkeypatch):
    """No keys at all → DuckDuckGo fallback."""
    for var in ("EXA_API_KEY", "TAVILY_API_KEY", "BRAVE_API_KEY", "SERPAPI_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    backend = auto_detect_backend()
    assert isinstance(backend, DuckDuckGoBackend)
