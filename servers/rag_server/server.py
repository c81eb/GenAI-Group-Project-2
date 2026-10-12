"""Custom RAG MCP server over the RAGAs docs. Exposes one tool: rag_search(query, k)."""
from mcp.server.mcpserver import MCPServer

from . import store

MAX_CHUNK_CHARS = 1200   # keep tool output small: it goes straight into the LLM's context

app = MCPServer("rag")


@app.tool()
def rag_search(query: str, k: int = 4) -> str:
    """Search the RAGAs documentation (RAG evaluation library: metrics, test-set generation,
    evaluation, how-tos). Use for any question about RAGAs. Returns the top-k passages."""
    k = max(1, min(k, 10))
    try:
        hits = store.search(query, k=k, rerank=True)
    except RuntimeError as e:   # index not built yet
        return str(e)
    return "\n\n---\n\n".join(
        f"[{i}] {h['source']} :: {h['heading']} (score {h['score']:.2f})\n{h['text'][:MAX_CHUNK_CHARS]}"
        for i, h in enumerate(hits, 1)
    )


if __name__ == "__main__":
    app.run()
