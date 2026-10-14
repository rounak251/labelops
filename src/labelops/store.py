"""Append-only annotation audit with current-vote and consensus views."""

from __future__ import annotations

import csv
import io
import itertools
import sqlite3
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Iterator


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _votes_to_consensus(annotations: list[dict]) -> dict[int, int]:
    labels: dict[int, list[int]] = defaultdict(list)
    for row in annotations:
        labels[row["sample_id"]].append(row["label"])
    consensus = {}
    for sample_id, votes in labels.items():
        if len(votes) >= 2:
            label, count = Counter(votes).most_common(1)[0]
            if count > len(votes) / 2:
                consensus[sample_id] = label
    return consensus


def _spreadsheet_text(value: str) -> str:
    """Neutralize formulas in exported free text without changing the audit."""
    if value.startswith(("=", "+", "-", "@", "\t", "\r")) or value.lstrip().startswith(
        ("=", "+", "-", "@")
    ):
        return "'" + value
    return value


class AnnotationStore:
    """Local SQLite event log; a revision adds an event and replaces a view vote."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS annotation_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sample_id INTEGER NOT NULL CHECK(sample_id >= 0),
                    reviewer TEXT NOT NULL CHECK(length(reviewer) BETWEEN 1 AND 80),
                    label INTEGER NOT NULL CHECK(label BETWEEN 0 AND 9),
                    note TEXT NOT NULL CHECK(length(note) <= 2000),
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS annotation_vote_lookup
                    ON annotation_events(sample_id, reviewer, event_id);
                CREATE TRIGGER IF NOT EXISTS annotation_events_no_update
                    BEFORE UPDATE ON annotation_events
                    BEGIN SELECT RAISE(ABORT, 'annotation audit is append-only'); END;
                CREATE TRIGGER IF NOT EXISTS annotation_events_no_delete
                    BEFORE DELETE ON annotation_events
                    BEGIN SELECT RAISE(ABORT, 'annotation audit is append-only'); END;
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def record(self, sample_id: int, reviewer: str, label: int, note: str = "") -> dict:
        sample_id = _integer(sample_id, "sample_id", 0, 2**63 - 1)
        label = _integer(label, "label", 0, 9)
        if not isinstance(reviewer, str):
            raise ValueError("reviewer must be text")
        reviewer = reviewer.strip()
        if not 1 <= len(reviewer) <= 80 or any(
            ord(character) < 32 or ord(character) == 127 for character in reviewer
        ):
            raise ValueError("reviewer must contain 1 to 80 printable characters")
        if not isinstance(note, str) or len(note) > 2000 or "\x00" in note:
            raise ValueError("note must be text of at most 2000 characters without NUL")
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO annotation_events(sample_id, reviewer, label, note, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (sample_id, reviewer, label, note, created_at),
            )
            event_id = cursor.lastrowid
        return {
            "event_id": event_id,
            "sample_id": sample_id,
            "reviewer": reviewer,
            "label": label,
            "note": note,
            "created_at": created_at,
        }

    def _snapshot(self) -> tuple[list[dict], int]:
        with self._connect() as connection:
            connection.execute("BEGIN")
            rows = connection.execute(
                "SELECT events.* FROM annotation_events AS events "
                "JOIN (SELECT MAX(event_id) AS latest FROM annotation_events "
                "GROUP BY sample_id, reviewer) AS current ON events.event_id = current.latest "
                "ORDER BY events.sample_id, events.reviewer"
            ).fetchall()
            event_count = connection.execute("SELECT COUNT(*) FROM annotation_events").fetchone()[0]
        return [dict(row) for row in rows], event_count

    def annotations(self) -> list[dict]:
        """Return one latest vote per sample and reviewer, in stable order."""
        return self._snapshot()[0]

    def events(self) -> list[dict]:
        """Return the immutable audit, including superseded votes."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM annotation_events ORDER BY event_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def consensus(self) -> dict[int, int]:
        return _votes_to_consensus(self.annotations())

    def agreement(self) -> dict:
        annotations, event_count = self._snapshot()
        consensus = _votes_to_consensus(annotations)
        by_reviewer: dict[str, dict[int, int]] = defaultdict(dict)
        review_counts: Counter = Counter()
        for row in annotations:
            by_reviewer[row["reviewer"]][row["sample_id"]] = row["label"]
            review_counts[row["sample_id"]] += 1
        pairwise = []
        for first, second in itertools.combinations(sorted(by_reviewer), 2):
            shared = sorted(by_reviewer[first].keys() & by_reviewer[second].keys())
            count = len(shared)
            agreement_rate, kappa = None, None
            if count:
                first_labels = [by_reviewer[first][sample_id] for sample_id in shared]
                second_labels = [by_reviewer[second][sample_id] for sample_id in shared]
                agreement_rate = sum(a == b for a, b in zip(first_labels, second_labels)) / count
                first_counts, second_counts = Counter(first_labels), Counter(second_labels)
                expected = sum(first_counts[label] * second_counts[label] for label in range(10))
                expected /= count * count
                # Identical constant labels yield 0/0, so expose undefined as JSON null.
                if expected < 1:
                    kappa = (agreement_rate - expected) / (1 - expected)
            pairwise.append(
                {
                    "reviewer_a": first,
                    "reviewer_b": second,
                    "shared_samples": count,
                    "agreement_rate": agreement_rate,
                    "cohen_kappa": kappa,
                }
            )
        return {
            "annotator_count": len(by_reviewer),
            "annotation_count": len(annotations),
            "event_count": event_count,
            "consensus_count": len(consensus),
            "disputed_count": sum(
                count >= 2 and sample_id not in consensus
                for sample_id, count in review_counts.items()
            ),
            "pairwise_kappa": pairwise,
        }

    def export_csv(self) -> str:
        """Export current votes; free text is escaped for spreadsheet readers."""
        output = io.StringIO(newline="")
        writer = csv.DictWriter(
            output, fieldnames=["sample_id", "reviewer", "label", "note", "created_at", "event_id"]
        )
        writer.writeheader()
        for annotation in self.annotations():
            row = dict(annotation)
            row["reviewer"] = _spreadsheet_text(row["reviewer"])
            row["note"] = _spreadsheet_text(row["note"])
            writer.writerow(row)
        return output.getvalue()
