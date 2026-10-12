"""Which MCP servers CodeMax starts: filesystem, Context7 and the custom RAG server."""
import os
import sys
from pathlib import Path

from .client import ServerConfig


# Filesystem tools that duplicate others (read_text_file ~ read_file) or are rarely needed.
FILESYSTEM_EXCLUDE = frozenset({
    "read_text_file", "read_media_file", "read_multiple_files", "list_directory_with_sizes",
    "directory_tree", "get_file_info", "list_allowed_directories",
})


def default_servers(workdir: str) -> list[ServerConfig]:
    servers = [
        # Official filesystem server, limited to the project directory.
        ServerConfig("filesystem", "npx",
                     ["-y", "@modelcontextprotocol/server-filesystem", workdir],
                     exclude=FILESYSTEM_EXCLUDE),
        # Context7: up-to-date library documentation lookup.
        ServerConfig("context7", "npx", ["-y", "@upstash/context7-mcp"],
                     env=_context7_env()),
        # Custom RAG over the RAGAs docs. Needs a one-time `python -m servers.rag_server.ingest`.
        ServerConfig("rag", sys.executable, ["-m", "servers.rag_server.server"],
                     env=_rag_env()),
    ]
    return servers


def _context7_env() -> dict[str, str]:
    # Stdio servers get a minimal environment, so pass PATH (for node) explicitly.
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "")}
    if os.getenv("CONTEXT7_API_KEY"):  # optional: raises rate limits
        env["CONTEXT7_API_KEY"] = os.environ["CONTEXT7_API_KEY"]
    return env


def _rag_env() -> dict[str, str]:
    project_root = str(Path(__file__).resolve().parents[2])  # so `-m servers.rag_server...` imports
    return {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", ""),
            "PYTHONPATH": project_root,
            "HF_HUB_OFFLINE": "1",            # models were cached at ingest; skip network checks
            "TOKENIZERS_PARALLELISM": "false"}
