"""Split markdown docs into chunks that keep their heading context.

Strategy: cut at markdown headings first (a section is one idea), then split any
long section into overlapping windows on paragraph boundaries. Each chunk is
prefixed with its heading path ("Page > Section > Sub") so it still makes sense
when retrieved on its own.
"""
import re
from dataclasses import dataclass

MAX_CHARS = 1200      # ~300 tokens: fits the embedding model's input comfortably
OVERLAP_CHARS = 150   # carry-over between windows of one long section
MIN_CHARS = 80        # drop near-empty chunks (e.g. a heading with no body)

_HEADING = re.compile(r"^(#{1,4})\s+(.*)$")


@dataclass
class Chunk:
    text: str       # heading path + body, this is what gets embedded
    source: str     # doc path relative to the docs root
    heading: str    # heading path only, shown in search results


def _sections(markdown: str, title: str):
    """Yield (heading_path, body) for each heading-delimited section."""
    path: list[str] = [title]
    body: list[str] = []
    in_code = False
    for line in markdown.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code            # '#' inside code fences is a comment, not a heading
        m = None if in_code else _HEADING.match(line)
        if m:
            if body:
                yield " > ".join(path), "\n".join(body).strip()
            level = len(m.group(1))
            path = path[:level]              # path[0] is the page title; deeper headings replace siblings
            path.append(m.group(2).strip())
            body = []
        else:
            body.append(line)
    if body:
        yield " > ".join(path), "\n".join(body).strip()


def _windows(body: str):
    """Split a long body on paragraph boundaries into <= MAX_CHARS windows with overlap."""
    if len(body) <= MAX_CHARS:
        yield body
        return
    current = ""
    for para in re.split(r"\n\s*\n", body):
        if current and len(current) + len(para) + 2 > MAX_CHARS:
            yield current
            current = current[-OVERLAP_CHARS:]
        current = f"{current}\n\n{para}" if current else para
        while len(current) > MAX_CHARS * 2:   # a single huge paragraph (e.g. a code block)
            yield current[:MAX_CHARS]
            current = current[MAX_CHARS - OVERLAP_CHARS:]
    if current.strip():
        yield current


def chunk_markdown(markdown: str, source: str) -> list[Chunk]:
    title = source.rsplit("/", 1)[-1].removesuffix(".md").replace("_", " ")
    chunks = []
    for heading, body in _sections(markdown, title):
        for window in _windows(body):
            if len(window.strip()) < MIN_CHARS:
                continue
            chunks.append(Chunk(f"{heading}\n\n{window.strip()}", source, heading))
    return chunks
