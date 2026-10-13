from __future__ import annotations

from enum import StrEnum

import httpx
import pytest
import respx
from click.testing import CliRunner

from vo_rag.cli import query as query_cli
from vo_rag.cli.main import cli
from vo_rag.core.answer.context import ContextScope
from vo_rag.core.answer.synthesizer import AnswerProfile
from vo_rag.core.retrieval.query_mode import QueryMode


@pytest.mark.parametrize(
    ("cli_choices", "enum_type"),
    [
        (query_cli._QUERY_MODES, QueryMode),
        (query_cli._ANSWER_PROFILES, AnswerProfile),
        (query_cli._CONTEXT_SCOPES, ContextScope),
    ],
)
def test_query_answer_choice_literals_match_core_enums(
    cli_choices: list[str], enum_type: type[StrEnum]
) -> None:
    assert cli_choices == [member.value for member in enum_type]


def test_query_posts_to_api(monkeypatch) -> None:
    monkeypatch.setenv("VORAG_API_KEY", "test-key")
    with respx.mock:
        route = respx.post("http://localhost:8000/api/v1/query").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": "chunk-1",
                            "score": 0.95,
                            "collection": "vo_rag_books_raw_v1",
                            "payload": {"text": "Sample text content"},
                        }
                    ]
                },
            )
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["query", "test query"])
        assert result.exit_code == 0
        assert route.called
        sent = route.calls[0].request.content
        assert b"test query" in sent
        assert "chunk-1" in result.output
        assert "0.9500" in result.output


def test_query_json_output(monkeypatch) -> None:
    monkeypatch.setenv("VORAG_API_KEY", "test-key")
    with respx.mock:
        respx.post("http://localhost:8000/api/v1/query").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["query", "test", "--format", "json"])
        assert result.exit_code == 0
        assert '"results": []' in result.output


def test_query_with_options(monkeypatch) -> None:
    monkeypatch.setenv("VORAG_API_KEY", "test-key")
    with respx.mock:
        route = respx.post("http://localhost:8000/api/v1/query").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "query",
                "test",
                "--collection",
                "vo_rag_books_raw_v1",
                "--top-k",
                "5",
                "--no-rerank",
                "--mode",
                "publication",
                "--visual-mode",
                "baseline",
            ],
        )
        assert result.exit_code == 0
        assert route.called
        sent = route.calls[0].request.content.decode()
        assert "vo_rag_books_raw_v1" in sent
        assert '"top_k":5' in sent
        assert '"rerank":false' in sent
        assert "publication" in sent
        assert "baseline" in sent


def test_answer_posts_to_api(monkeypatch) -> None:
    monkeypatch.setenv("VORAG_API_KEY", "test-key")
    with respx.mock:
        route = respx.post("http://localhost:8000/api/v1/answer").mock(
            return_value=httpx.Response(
                200,
                json={
                    "answer_id": "ans-1",
                    "answer": "The answer is 42.",
                    "confidence": 0.95,
                    "citations": ["source-1"],
                    "sources": ["source-1"],
                    "retrieval_results": [],
                    "compliance": {"should_quarantine": False},
                    "warnings": [],
                    "model": "claude-3-sonnet",
                    "mode": "internal",
                },
            )
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["answer", "what is the meaning?"])
        assert result.exit_code == 0
        assert route.called
        sent = route.calls[0].request.content
        assert b"what is the meaning?" in sent
        assert "The answer is 42." in result.output


def test_answer_json_output(monkeypatch) -> None:
    monkeypatch.setenv("VORAG_API_KEY", "test-key")
    with respx.mock:
        respx.post("http://localhost:8000/api/v1/answer").mock(
            return_value=httpx.Response(
                200,
                json={
                    "answer_id": "ans-1",
                    "answer": "Yes.",
                    "confidence": 0.8,
                    "citations": [],
                    "sources": [],
                    "retrieval_results": [],
                    "compliance": {"should_quarantine": False},
                    "warnings": ["low confidence"],
                    "model": "claude-3-sonnet",
                    "mode": "internal",
                },
            )
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["answer", "test", "--format", "json"])
        assert result.exit_code == 0
        assert "Yes." in result.output
        assert '"answer_id":' in result.output


def test_answer_with_all_options(monkeypatch) -> None:
    monkeypatch.setenv("VORAG_API_KEY", "test-key")
    with respx.mock:
        route = respx.post("http://localhost:8000/api/v1/answer").mock(
            return_value=httpx.Response(
                200,
                json={
                    "answer_id": "ans-1",
                    "answer": "Yes.",
                    "confidence": 0.8,
                    "citations": [],
                    "sources": [],
                    "retrieval_results": [],
                    "compliance": {"should_quarantine": False},
                    "warnings": [],
                    "model": "claude-3-sonnet",
                    "mode": "internal",
                },
            )
        )
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "answer",
                "test",
                "--profile",
                "fast",
                "--context-scope",
                "raw",
                "--no-include-retrieval",
                "--mode",
                "partner",
                "--collection",
                "vo_rag_articles_v1",
                "--top-k",
                "3",
                "--no-rerank",
                "--visual-mode",
                "visual_only",
            ],
        )
        assert result.exit_code == 0
        assert route.called
        sent = route.calls[0].request.content.decode()
        assert "fast" in sent
        assert "raw" in sent
        assert '"include_retrieval":false' in sent
        assert "partner" in sent
        assert "vo_rag_articles_v1" in sent
        assert '"top_k":3' in sent
        assert '"rerank":false' in sent
        assert "visual_only" in sent
