from pathlib import Path

from src.feature_research import CandidateFeatures, load_candidates, record_proposals, save_candidates


def test_load_candidates_missing_file_returns_empty(tmp_path: Path):
    doc = load_candidates(tmp_path / "does_not_exist.yaml")
    assert doc.candidates == []
    assert doc.research_history == []


def test_save_and_load_roundtrip(tmp_path: Path):
    doc = CandidateFeatures(
        candidates=[{"id": "foo", "name": "Foo", "status": "shadow"}],
        research_history=[{"date": "2026-01-01", "n_proposed": 1, "n_added": 1}],
    )
    path = tmp_path / "candidates.yaml"
    save_candidates(doc, path)
    loaded = load_candidates(path)

    assert loaded.candidates == doc.candidates
    assert loaded.research_history == doc.research_history


def test_record_proposals_adds_new_and_skips_duplicates():
    doc = CandidateFeatures(candidates=[{"id": "existing", "status": "shadow"}])
    proposals = [
        {"id": "existing", "name": "dup", "description": "...", "rationale": "..."},
        {"id": "new_one", "name": "New", "description": "...", "rationale": "..."},
    ]

    added = record_proposals(doc, proposals)

    assert added == 1
    ids = {c["id"] for c in doc.candidates}
    assert ids == {"existing", "new_one"}
    new_candidate = next(c for c in doc.candidates if c["id"] == "new_one")
    assert new_candidate["status"] == "proposed"
    assert len(doc.research_history) == 1
    assert doc.research_history[0]["n_added"] == 1
