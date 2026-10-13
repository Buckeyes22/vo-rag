"""`vo-rag corpus` — in-process corpus maintenance. Operator-only.

Runs against Postgres / Garage from settings like `vo-rag ingest run`. Prints
one JSON object per line to stdout; exit code 1 on any error line.

Other modules extend the group through ``corpus_group()`` (they add
``categories``, ``purge``, ``duplicates``, ``health``). This file holds the
group only.
"""

from __future__ import annotations

import click

_GROUP: click.Group | None = None


def corpus_group() -> click.Group:
    if _GROUP is None:
        raise RuntimeError("register_corpus_commands must run first")
    return _GROUP


def register_corpus_commands(root: click.Group) -> None:
    global _GROUP

    @root.group("corpus")
    def corpus() -> None:
        """Corpus maintenance: OCR farm for scanned books."""

    _GROUP = corpus
    from vo_rag.cli import corpus_ocr, corpus_ops, corpus_rechunk, corpus_tf

    corpus_ocr.register(corpus)
    corpus_ops.register(corpus)
    corpus_rechunk.register(corpus)
    corpus_tf.register(corpus)
