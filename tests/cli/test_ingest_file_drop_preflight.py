"""``vo-rag ingest file-drop --preflight report.json`` (VOR-1434).

TARGETS become local file paths (uploaded exactly like ``ingest file``), each
matched against a ``vo-rag preflight --out report.json`` report by file
path, else by the sha256 computed during upload, and its
``metadata.preflight = {route, pages, kind, verdict, sha256}`` attached to
the submitted envelope. A file the report marks ``reject`` is skipped
(non-zero exit) unless ``--allow-rejects``.

Legacy ``ingest file-drop <garage_key>`` (no ``--preflight``) is unchanged —
covered by ``test_ingest_file_drop_posts_to_api`` in
``test_cli_ingest_eval_admin.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import respx
from click.testing import CliRunner

from vo_rag.cli.main import cli

_CONTENT = b"a giant book, allegedly"


def _write_report(tmp_path: Path, results: list[dict[str, object]]) -> Path:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"counts": {}, "duplicate_groups": [], "results": results}),
        encoding="utf-8",
    )
    return report


def _mock_upload(garage_key: str) -> None:
    respx.post("http://localhost:8000/api/v1/intake/upload-url").mock(
        return_value=httpx.Response(
            200,
            json={
                "upload_url": "https://storage.example.com/upload",
                "garage_key": garage_key,
                "bucket": "test-bucket",
                "expires_in_s": 3600,
            },
        )
    )
    respx.put("https://storage.example.com/upload").mock(return_value=httpx.Response(200))


def test_preflight_metadata_attached_when_matched_by_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VORAG_API_KEY", "test-key")
    book = tmp_path / "book.pdf"
    book.write_bytes(_CONTENT)
    report = _write_report(
        tmp_path,
        [
            {
                "path": str(book),
                "sha256": "irrelevant-since-matched-by-path",
                "kind": "pdf",
                "pages": 1200,
                "route": "giant",
                "verdict": "ok",
                "reasons": [],
            }
        ],
    )

    with respx.mock:
        _mock_upload("gk-giant-1")
        intake_route = respx.post("http://localhost:8000/api/v1/intake").mock(
            return_value=httpx.Response(200, json={"envelope_id": "env-giant"})
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["ingest", "file-drop", str(book), "--preflight", str(report)])
        assert result.exit_code == 0, result.output
        payload = json.loads(intake_route.calls[0].request.content.decode())
        assert payload["metadata"]["preflight"] == {
            "route": "giant",
            "pages": 1200,
            "kind": "pdf",
            "verdict": "ok",
            "sha256": "irrelevant-since-matched-by-path",
        }
        assert "env-giant" in result.output


def test_preflight_metadata_attached_when_matched_by_sha256_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    """The report used a different path spelling for the same file -- the
    match falls back to the sha256 computed during upload."""
    monkeypatch.setenv("VORAG_API_KEY", "test-key")
    book = tmp_path / "book.pdf"
    book.write_bytes(_CONTENT)
    import hashlib

    sha256 = hashlib.sha256(_CONTENT).hexdigest()
    report = _write_report(
        tmp_path,
        [
            {
                "path": "/some/other/machine/path/book.pdf",
                "sha256": sha256,
                "kind": "pdf",
                "pages": 42,
                "route": "native",
                "verdict": "ok",
                "reasons": [],
            }
        ],
    )

    with respx.mock:
        _mock_upload("gk-native-1")
        intake_route = respx.post("http://localhost:8000/api/v1/intake").mock(
            return_value=httpx.Response(200, json={"envelope_id": "env-native"})
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["ingest", "file-drop", str(book), "--preflight", str(report)])
        assert result.exit_code == 0, result.output
        payload = json.loads(intake_route.calls[0].request.content.decode())
        assert payload["metadata"]["preflight"]["route"] == "native"
        assert payload["metadata"]["preflight"]["sha256"] == sha256


def test_reject_verdict_is_skipped_and_exits_non_zero(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VORAG_API_KEY", "test-key")
    book = tmp_path / "flyer.pdf"
    book.write_bytes(_CONTENT)
    report = _write_report(
        tmp_path,
        [
            {
                "path": str(book),
                "sha256": "x" * 64,
                "kind": "pdf",
                "pages": 2,
                "route": None,
                "verdict": "reject",
                "reasons": ["not_a_book"],
            }
        ],
    )

    with respx.mock:
        intake_route = respx.post("http://localhost:8000/api/v1/intake").mock(
            return_value=httpx.Response(200, json={"envelope_id": "should-not-be-called"})
        )
        upload_route = respx.post("http://localhost:8000/api/v1/intake/upload-url")
        runner = CliRunner()
        result = runner.invoke(cli, ["ingest", "file-drop", str(book), "--preflight", str(report)])
        assert result.exit_code != 0
        assert "skipping" in result.output
        assert "reject" in result.output
        assert not intake_route.called
        assert not upload_route.called


def test_allow_rejects_submits_the_rejected_file_anyway(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VORAG_API_KEY", "test-key")
    book = tmp_path / "flyer.pdf"
    book.write_bytes(_CONTENT)
    report = _write_report(
        tmp_path,
        [
            {
                "path": str(book),
                "sha256": "x" * 64,
                "kind": "pdf",
                "pages": 2,
                "route": None,
                "verdict": "reject",
                "reasons": ["not_a_book"],
            }
        ],
    )

    with respx.mock:
        _mock_upload("gk-flyer-1")
        intake_route = respx.post("http://localhost:8000/api/v1/intake").mock(
            return_value=httpx.Response(200, json={"envelope_id": "env-flyer"})
        )
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "ingest",
                "file-drop",
                str(book),
                "--preflight",
                str(report),
                "--allow-rejects",
            ],
        )
        assert result.exit_code == 0, result.output
        assert intake_route.called
        payload = json.loads(intake_route.calls[0].request.content.decode())
        assert payload["metadata"]["preflight"]["verdict"] == "reject"


def test_mixed_batch_submits_ok_files_and_skips_rejects(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VORAG_API_KEY", "test-key")
    ok_book = tmp_path / "ok.pdf"
    ok_book.write_bytes(_CONTENT)
    reject_book = tmp_path / "reject.pdf"
    reject_book.write_bytes(_CONTENT + b"more")
    report = _write_report(
        tmp_path,
        [
            {
                "path": str(ok_book),
                "sha256": "a" * 64,
                "kind": "pdf",
                "pages": 10,
                "route": "native",
                "verdict": "ok",
                "reasons": [],
            },
            {
                "path": str(reject_book),
                "sha256": "b" * 64,
                "kind": "pdf",
                "pages": 3,
                "route": None,
                "verdict": "reject",
                "reasons": ["not_a_book"],
            },
        ],
    )

    with respx.mock:
        _mock_upload("gk-mixed-1")
        intake_route = respx.post("http://localhost:8000/api/v1/intake").mock(
            return_value=httpx.Response(200, json={"envelope_id": "env-ok"})
        )
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["ingest", "file-drop", str(ok_book), str(reject_book), "--preflight", str(report)],
        )
        assert result.exit_code != 0
        assert intake_route.call_count == 1
        assert "env-ok" in result.output
        assert "skipping" in result.output


def test_files_absent_from_the_report_are_submitted_without_preflight_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VORAG_API_KEY", "test-key")
    book = tmp_path / "not_in_report.pdf"
    book.write_bytes(_CONTENT)
    report = _write_report(tmp_path, [])

    with respx.mock:
        _mock_upload("gk-unmatched-1")
        intake_route = respx.post("http://localhost:8000/api/v1/intake").mock(
            return_value=httpx.Response(200, json={"envelope_id": "env-unmatched"})
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["ingest", "file-drop", str(book), "--preflight", str(report)])
        assert result.exit_code == 0, result.output
        payload = json.loads(intake_route.calls[0].request.content.decode())
        assert "preflight" not in payload["metadata"]
