"""Tests for the RAG server's chunker (no models or DB needed)."""
from servers.rag_server.chunking import MAX_CHARS, chunk_markdown

DOC = """# Faithfulness

Faithfulness measures how factually consistent a response is with the retrieved context. It ranges from 0 to 1.

## Example

```python
# this hash is a comment, not a heading
score = metric.score(sample)
```

Higher scores mean fewer hallucinated claims in the generated answer text.
"""


def test_chunks_keep_heading_path():
    chunks = chunk_markdown(DOC, "metrics/faithfulness.md")
    assert chunks and all(c.source == "metrics/faithfulness.md" for c in chunks)
    assert any(c.heading == "faithfulness > Faithfulness > Example" for c in chunks)
    assert all(c.text.startswith(c.heading) for c in chunks)


def test_hash_in_code_fence_is_not_a_heading():
    chunks = chunk_markdown(DOC, "m.md")
    assert not any("this hash is a comment" in c.heading for c in chunks)


def test_long_section_is_split_with_bounded_size():
    body = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(30))
    chunks = chunk_markdown("# Big\n\n" + body, "big.md")
    assert len(chunks) > 1
    assert all(len(c.text) <= MAX_CHARS * 2 for c in chunks)


def test_tiny_sections_are_dropped():
    assert chunk_markdown("# A\n\nhi\n", "a.md") == []


def test_rag_search_over_mcp():
    """End to end through the real MCP client. Skipped if the index has not been built."""
    import os
    import pytest
    from servers.rag_server import store
    if store.collection().count() == 0:
        pytest.skip("run `python -m servers.rag_server.ingest` first")
    from codemax.mcp_client.config import default_servers
    from codemax.mcp_client.client import MCPClient
    client = MCPClient([s for s in default_servers(os.getcwd()) if s.name == "rag"])
    try:
        tools = client.connect()
        assert [t.name for t in tools] == ["rag__rag_search"], client.errors
        out = client.call_tool("rag__rag_search", {"query": "What is context recall?", "k": 2})
        assert "context_recall" in out
    finally:
        client.close()
