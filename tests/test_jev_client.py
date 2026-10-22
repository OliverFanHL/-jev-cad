import logging
from unittest.mock import patch

import pandas as pd

from jev_tmcad.models import jev as jev_client


def test_csv_missing_extract_error_is_not_skipped(tmp_path, monkeypatch, caplog):
    csv_path = tmp_path / "descriptors.csv"
    pd.DataFrame(
        [
            {"sample_id": "valid", "label": "gear", "split": "test", "absolute_path": "valid.stp", "extract_error": ""},
            {"sample_id": "failed", "label": "gear", "split": "test", "absolute_path": "failed.stp", "extract_error": "ValueError: STEP File could not be loaded"},
        ]
    ).to_csv(csv_path, index=False)
    rows = pd.read_csv(csv_path)
    monkeypatch.setattr(jev_client, "build_semantic_state", lambda row, **kwargs: {"valid": True})

    class Schema:
        def to_dict(self):
            return {}

    dry_path = tmp_path / "states.jsonl"
    with caplog.at_level(logging.WARNING):
        jev_client.run_jev_for_dataframe(
            rows, Schema(), {"classification": {}}, tmp_path / "results.jsonl",
            tmp_path / "cache.sqlite", dry_run_states=dry_path,
        )

    assert [r["sample_id"] for r in jev_client.read_jsonl(dry_path)] == ["valid"]
    assert "failed.stp: ValueError: STEP File could not be loaded" in caplog.text
    assert "Skipping sample valid" not in caplog.text


def test_dry_run_counts_pending_requests_without_duplicating_preview(tmp_path):
    rows = pd.DataFrame([{"sample_id": "part-a", "label": "gear", "split": "test",
                          "relative_path": "gear/a.stp", "absolute_path": "a.stp",
                          "extract_error": ""}])

    class Schema:
        def to_dict(self):
            return {}

    dry_path = tmp_path / "states.jsonl"
    with patch.object(jev_client, "build_semantic_state", return_value={"geometry": {"x": 1}}):
        first, second = {}, {}
        for stats in (first, second):
            result = jev_client.run_jev_for_dataframe(
                rows, Schema(), {"classification": {}}, tmp_path / "results.jsonl",
                tmp_path / "cache.sqlite", dry_run_states=dry_path,
                dry_run_stats=stats,
            )
            assert result == []
    assert first["pending_responses"] == second["pending_responses"] == 1
    assert first["uncached_requests"] == second["uncached_requests"] == 1
    assert first["previews_added"] == 1
    assert second["previews_added"] == 0
    assert len(jev_client.read_jsonl(dry_path)) == 1
