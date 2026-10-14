"""Voting invariants, persistent revisions, and safe agreement/export behavior."""

import csv
import io
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from labelops.store import AnnotationStore


def test_revision_preserves_audit_and_one_latest_vote(tmp_path):
    store = AnnotationStore(tmp_path / "annotations.sqlite3")
    first = store.record(12, "Ada", 1, "initial")
    store.record(12, "Grace", 1)
    assert store.consensus() == {12: 1}
    revised = store.record(12, "Ada", 2, "reconsidered")
    assert revised["event_id"] > first["event_id"]
    assert store.consensus() == {}
    assert len(store.annotations()) == 2
    assert len(store.events()) == 3
    assert store.events()[0]["note"] == "initial"
    assert store.agreement()["disputed_count"] == 1
    reopened = AnnotationStore(store.path)
    assert reopened.annotations() == store.annotations()
    assert reopened.events() == store.events()
    with sqlite3.connect(store.path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE annotation_events SET label = 4")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM annotation_events")


def test_consensus_requires_distinct_reviewers_and_strict_majority(tmp_path):
    store = AnnotationStore(tmp_path / "votes.sqlite3")
    store.record(3, "Ada", 4)
    store.record(3, "Ada", 4)
    assert store.consensus() == {}
    store.record(3, "Grace", 8)
    assert store.consensus() == {}
    store.record(3, "Alan", 4)
    assert store.consensus() == {3: 4}
    store.record(3, "Katherine", 8)
    assert store.consensus() == {}
    assert store.agreement()["annotation_count"] == 4
    assert store.agreement()["event_count"] == 5


def test_pairwise_agreement_uses_latest_overlapping_votes(tmp_path):
    store = AnnotationStore(tmp_path / "votes.sqlite3")
    for sample_id, first, second in ((1, 0, 0), (2, 1, 1), (3, 0, 1), (4, 1, 0)):
        store.record(sample_id, "Ada", first)
        store.record(sample_id, "Grace", second)
    store.record(99, "Ada", 9)
    pair = store.agreement()["pairwise_kappa"][0]
    assert pair["shared_samples"] == 4
    assert pair["agreement_rate"] == 0.5
    assert pair["cohen_kappa"] == 0.0
    store.record(3, "Grace", 0)
    assert store.agreement()["pairwise_kappa"][0]["cohen_kappa"] == 0.5


def test_undefined_kappa_is_json_null(tmp_path):
    store = AnnotationStore(tmp_path / "votes.sqlite3")
    store.record(1, "Ada", 3)
    store.record(1, "Grace", 3)
    store.record(2, "Alan", 6)
    result = store.agreement()
    assert all(pair["cohen_kappa"] is None for pair in result["pairwise_kappa"])
    assert result["consensus_count"] == 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "sample_id,reviewer,label,note",
    [
        (True, "Ada", 1, ""),
        (-1, "Ada", 1, ""),
        (2.5, "Ada", 1, ""),
        (1, "Ada", True, ""),
        (1, "Ada", 10, ""),
        (1, "Ada", -1, ""),
        (1, "", 1, ""),
        (1, "   ", 1, ""),
        (1, "A" * 81, 1, ""),
        (1, "Ada\nGrace", 1, ""),
        (1, None, 1, ""),
        (1, "Ada", 1, "x" * 2001),
        (1, "Ada", 1, "\x00"),
        (1, "Ada", 1, None),
    ],
)
def test_invalid_votes_never_create_events(tmp_path, sample_id, reviewer, label, note):
    store = AnnotationStore(tmp_path / "votes.sqlite3")
    with pytest.raises(ValueError):
        store.record(sample_id, reviewer, label, note)
    assert store.events() == []


def test_csv_preserves_latest_votes_and_neutralizes_formulas(tmp_path):
    store = AnnotationStore(tmp_path / "votes.sqlite3")
    store.record(1, "=SUM(1,2)", 3, "initial")
    store.record(1, "=SUM(1,2)", 4, '\t=HYPERLINK("example")')
    store.record(2, "Ada", 2, "ordinary\ntext, with comma")
    exported = list(csv.DictReader(io.StringIO(store.export_csv())))
    assert len(exported) == 2
    assert exported[0]["reviewer"] == "'=SUM(1,2)"
    assert exported[0]["note"].startswith("'\t=")
    assert exported[0]["label"] == "4"
    assert exported[1]["note"] == "ordinary\ntext, with comma"
    assert store.annotations()[0]["reviewer"] == "=SUM(1,2)"
    assert store.annotations()[0]["note"].startswith("\t=")


def test_store_connections_work_from_different_threads(tmp_path):
    store = AnnotationStore(tmp_path / "votes.sqlite3")
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda index: store.record(7, f"reviewer-{index}", 5), range(12)))
    assert store.consensus() == {7: 5}
    assert len(store.events()) == 12
