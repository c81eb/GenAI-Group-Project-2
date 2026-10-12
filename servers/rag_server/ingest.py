"""One-time ingest: load RAGAs docs, chunk, embed, store in Chroma.

    python -m servers.rag_server.ingest            # fetch docs from GitHub if needed, then index
    python -m servers.rag_server.ingest --docs-dir PATH   # index a local folder of .md files
    python -m servers.rag_server.ingest --force    # rebuild even if an index exists

Later sessions just reuse the persisted DB, so this only has to run once.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from .chunking import chunk_markdown
from .store import DB_DIR, collection, embed, reset_collection

REPO = "https://github.com/explodinggradients/ragas.git"
DOCS_CACHE = Path(__file__).parent / "data"   # gitignored checkout of the RAGAs docs
BATCH = 64


def fetch_docs() -> Path:
    """Shallow, docs-only checkout of the RAGAs repo (skips code and history)."""
    repo = DOCS_CACHE / "ragas"
    if not repo.exists():
        DOCS_CACHE.mkdir(parents=True, exist_ok=True)
        print(f"Cloning RAGAs docs from {REPO} ...")
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse",
                        REPO, str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", "docs"], check=True)
    return repo / "docs"


def load_chunks(docs_dir: Path):
    chunks = []
    files = sorted(docs_dir.rglob("*.md"))
    for f in files:
        rel = f.relative_to(docs_dir).as_posix()
        chunks += chunk_markdown(f.read_text(encoding="utf-8", errors="ignore"), rel)
    print(f"Loaded {len(files)} markdown files -> {len(chunks)} chunks")
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs-dir", type=Path, help="local folder of .md files (default: fetch RAGAs docs)")
    ap.add_argument("--force", action="store_true", help="rebuild an existing index")
    args = ap.parse_args()

    existing = collection().count()
    if existing and not args.force:
        print(f"Index already has {existing} chunks at {DB_DIR}. Use --force to rebuild.")
        return 0

    docs_dir = args.docs_dir or fetch_docs()
    chunks = load_chunks(docs_dir)
    if not chunks:
        print("No chunks found - nothing to index.", file=sys.stderr)
        return 1

    col = reset_collection()
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        col.add(
            ids=[f"chunk-{i + j}" for j in range(len(batch))],
            documents=[c.text for c in batch],
            embeddings=embed([c.text for c in batch]),
            metadatas=[{"source": c.source, "heading": c.heading} for c in batch],
        )
        print(f"  indexed {min(i + BATCH, len(chunks))}/{len(chunks)}", end="\r")
    print(f"\nDone. {col.count()} chunks stored in {DB_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
