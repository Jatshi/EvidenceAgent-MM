from __future__ import annotations

from evidenceagent_mm.provenance import huggingface_cache_revision


def test_huggingface_cache_revision_reads_main_ref(tmp_path) -> None:
    ref = tmp_path / "hub" / "models--org--model" / "refs" / "main"
    ref.parent.mkdir(parents=True)
    ref.write_text("abc123\n", encoding="utf-8")

    assert huggingface_cache_revision("org/model", tmp_path) == "abc123"


def test_huggingface_cache_revision_uses_only_unambiguous_snapshot(tmp_path) -> None:
    snapshots = tmp_path / "hub" / "models--org--model" / "snapshots"
    (snapshots / "only-revision").mkdir(parents=True)

    assert huggingface_cache_revision("org/model", tmp_path) == "only-revision"
    (snapshots / "another-revision").mkdir()
    assert huggingface_cache_revision("org/model", tmp_path) is None
