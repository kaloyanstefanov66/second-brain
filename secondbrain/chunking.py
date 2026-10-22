"""Split document sections into overlapping, citation-friendly chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .loaders import Section

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


@dataclass
class Chunk:
    text: str
    heading: str = ""
    page: int | None = None


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]


def _pieces(text: str, size: int) -> list[str]:
    """Break text into paragraph-ish pieces no longer than `size`."""
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            out.append(para)
            continue
        buf = ""
        for sent in split_sentences(para):
            while len(sent) > size:  # pathological: no sentence boundaries
                cut = sent.rfind(" ", 0, size)
                cut = cut if cut > size // 2 else size
                if buf:
                    out.append(buf)
                    buf = ""
                out.append(sent[:cut].strip())
                sent = sent[cut:].strip()
            if buf and len(buf) + len(sent) + 1 > size:
                out.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf:
            out.append(buf)
    return out


def _tail(text: str, overlap: int) -> str:
    """The last sentence(s) of text, up to `overlap` chars, to carry into the next chunk."""
    if overlap <= 0:
        return ""
    tail = ""
    for sent in reversed(split_sentences(text)):
        if len(tail) + len(sent) > overlap:
            break
        tail = f"{sent} {tail}".strip()
    return tail


def chunk_sections(sections: list[Section], size: int = 1000, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for sec in sections:
        buf = ""
        fresh = False  # buf holds new content beyond carried-over overlap
        for piece in _pieces(sec.text, size):
            if buf and len(buf) + len(piece) + 2 > size and fresh:
                chunks.append(Chunk(buf, sec.heading, sec.page))
                buf, fresh = _tail(buf, overlap), False
            buf = f"{buf}\n\n{piece}".strip() if buf else piece
            fresh = True
        if buf and fresh:
            chunks.append(Chunk(buf, sec.heading, sec.page))
    return chunks
