"""Filesystem figure-image store — NVMe origin for extracted figures.

Figure bytes live on local NVMe
under ``settings.figures.root`` addressed by a deterministic relative key

    {src}/p{page:03d}_img{index:03d}.{ext}

where ``src`` is sha256 of the source file bytes — the same canonical
identity already used for ``source_id`` (``vo-rag-corpus-sha256:<src>``).
Same source bytes + same page/figure index ⇒ the same key every run, so
re-ingest is idempotent and Qdrant ``image_ref`` payloads stay valid
across re-extraction.

No S3/Garage dependency: ``write`` / ``open_bytes`` / ``exists`` are plain
filesystem operations, trivially testable with ``tmp_path``. The serving
route (``api/routes/figures.py``) shares the validation tables below so
write-side and serve-side accept exactly the same key shapes.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import tempfile
from pathlib import Path

# ``:03d`` is MINIMUM width — books over 999 pages legitimately produce
# 4-digit page numbers, so validation accepts 3-5 digits. ``gif`` extends
# the spec's webp|png|jpg|jpeg set because EPUB embedded images are often
# GIFs; svg is deliberately excluded (script-in-svg XSS when served raw).
SRC_RE = re.compile(r"^[0-9a-f]{64}$")
FILENAME_RE = re.compile(
    r"^p\d{3,5}_(?:img\d{3,5}\.(?:webp|png|jpg|jpeg|gif)|(?:eq|tbl|pg)[0-9a-f]{64}\.png)$"
)
ALLOWED_EXTS = frozenset({"webp", "png", "jpg", "jpeg", "gif"})
MIME_BY_EXT = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
}
# MIME subtypes that are hard-refused even when the reported extension
# looks allowed: an EPUB item named ``cover.png`` carrying
# ``media_type=image/svg+xml`` must not slip SVG bytes under a png label.
REFUSED_MIME_SUBTYPES = frozenset({"svg", "svg+xml"})


def safe_ext(ext: str, mime_type: str) -> str | None:
    """Normalize a parser-reported extension to a storable/servable one.

    Returns ``None`` for types the store refuses to hold (svg, tiff, bmp,
    …) — callers skip those figures; the bytes remain regenerable from the
    source document. A refused MIME subtype wins over an allowed-looking
    extension (mislabeled-SVG defense).
    """
    subtype = (mime_type or "").split("/")[-1].lower()
    if subtype in REFUSED_MIME_SUBTYPES:
        return None
    candidate = (ext or "").lower().lstrip(".")
    if candidate == "jpe":
        candidate = "jpg"
    if candidate in ALLOWED_EXTS:
        return candidate
    guessed = (mime_type or "").split("/")[-1].lower()
    if guessed == "jpeg":
        guessed = "jpg"
    if guessed in ALLOWED_EXTS:
        return guessed
    return None


class FigureStore:
    """Deterministic-key filesystem store for extracted figure images."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    @staticmethod
    def figure_ref(src: str, page: int, index: int, ext: str) -> str:
        """Build the relative key; raises ValueError on malformed parts."""
        if not SRC_RE.match(src):
            raise ValueError(f"figure src must be 64 lowercase hex chars: {src!r}")
        filename = f"p{page:03d}_img{index:03d}.{ext}"
        if not FILENAME_RE.match(filename):
            raise ValueError(f"figure filename failed validation: {filename!r}")
        return f"{src}/{filename}"

    def path_for(self, ref: str) -> Path:
        """Resolve a relative key to an absolute path inside the root.

        Raises ValueError for malformed keys or anything that escapes the
        root (traversal). Never touches the filesystem.
        """
        src, _, filename = str(ref).partition("/")
        if not SRC_RE.match(src) or not FILENAME_RE.match(filename):
            raise ValueError(f"invalid figure ref: {ref!r}")
        path = (self._root / src / filename).resolve()
        if not path.is_relative_to(self._root):
            raise ValueError(f"figure ref escapes store root: {ref!r}")
        return path

    def write(self, src: str, page: int, index: int, data: bytes, *, ext: str) -> str:
        """Write figure bytes at the deterministic key; returns the key.

        Write-then-rename so a crash mid-write never leaves a partial file
        at a servable name (``FILENAME_RE`` rejects the ``.tmp`` suffix).
        """
        ref = self.figure_ref(src, page, index, ext)
        path = self.path_for(ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Unique temp name per writer: two parallel ingests of the same
        # source/page/index must not share a temp file (a shared name loses
        # one writer's replace()); os.replace keeps the final name atomic.
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp_name, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise
        return ref

    def open_bytes(self, ref: str) -> bytes:
        return self.path_for(ref).read_bytes()

    def write_equation(self, src: str, page: int, data: bytes) -> str:
        """Publish a PNG equation crop without replacing existing image bytes.

        The crop hash gets a separate key namespace from historical figure
        ordinals. Atomic link publication never exposes a partial image and
        never overwrites a conflicting existing file.
        """
        return self._write_source_crop(src, page, data, "eq")

    def write_table(self, src: str, page: int, data: bytes) -> str:
        """Publish original table pixels in an append-only content-addressed namespace."""
        return self._write_source_crop(src, page, data, "tbl")

    def write_page(self, src: str, page: int, data: bytes) -> str:
        """Publish original mathematical page pixels without replacing other figures."""
        return self._write_source_crop(src, page, data, "pg")

    def _write_source_crop(self, src: str, page: int, data: bytes, kind: str) -> str:
        digest = hashlib.sha256(data).hexdigest()
        ref = f"{src}/p{page:03d}_{kind}{digest}.png"
        path = self.path_for(ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
            try:
                os.link(temporary, path)
            except FileExistsError:
                with path.open("rb") as stream:
                    existing = stream.read(len(data) + 1)
                if existing != data:
                    raise ValueError("source crop key contains conflicting bytes") from None
        finally:
            os.unlink(temporary)
        return ref

    def exists(self, ref: str) -> bool:
        try:
            return self.path_for(ref).is_file()
        except ValueError:
            return False


__all__ = [
    "ALLOWED_EXTS",
    "FILENAME_RE",
    "MIME_BY_EXT",
    "SRC_RE",
    "FigureStore",
    "safe_ext",
]
