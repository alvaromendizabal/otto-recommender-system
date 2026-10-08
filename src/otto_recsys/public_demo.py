"""A small, reproducible session recommender using only the standard library.

The generated data are synthetic. The transparent weights below are illustrative
defaults, not tuned results. Historical session co-occurrence, observed recency,
and historical popularity are combined without using future targets. Predictions
are sealed to disk before the separate evaluation function receives those targets.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ACTIONS = ("clicks", "carts", "orders")
WEIGHTS = {"clicks": 0.1, "carts": 0.3, "orders": 0.6}
CONFIGURATION = {
    "history_sessions": 240,
    "query_sessions": 48,
    "catalog_items": 120,
    "k": 20,
    "candidate_limit": 50,
}
HISTORY_CUTOFF = 100_000
_OWNER = "otto-recsys-public-demo-v1"
_ARTIFACTS = ("predictions.json", "result.json", "report.html")


@dataclass(frozen=True)
class Event:
    """An anonymous item interaction at an integer synthetic timestamp."""

    session: int
    item: int
    ts: int
    action: str


@dataclass(frozen=True)
class Query:
    """The only session information available to inference."""

    session: int
    query_ts: int
    prefix: tuple[Event, ...]


@dataclass(frozen=True)
class DemoData:
    seed: int
    history: tuple[Event, ...]
    queries: tuple[Query, ...]
    targets: tuple[Event, ...]


@dataclass(frozen=True)
class DemoModel:
    """Counts fitted exclusively from historical sessions before ``cutoff``."""

    cutoff: int
    catalog: tuple[int, ...]
    popularity: dict[str, dict[int, int]]
    cooccurrence: dict[int, dict[int, int]]


def canonical_json(value: Any) -> bytes:
    """Stable UTF-8 JSON; NaN and infinity are deliberately rejected."""
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return (content + "\n").encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _event(event: Event) -> None:
    _integer(event.session, "session")
    _integer(event.item, "item")
    _integer(event.ts, "timestamp")
    if event.action not in ACTIONS:
        raise ValueError(f"Unknown action: {event.action}")


def generate_synthetic(seed: int = 42) -> DemoData:
    """Generate topic-based browsing with repeat intent and unrelated interactions.

    Independent random streams generate history, observed prefixes, and targets.
    Changing future-target generation therefore cannot change inference inputs.
    Integer timestamps express ordering only, not real customer activity.
    """
    _integer(seed, "seed")
    history_rng = random.Random(seed)
    query_rng = random.Random(seed + 1)
    target_rng = random.Random(seed + 2)
    history: list[Event] = []
    queries: list[Query] = []
    targets: list[Event] = []

    def item_for(rng: random.Random, topic: int, focus: float = 0.8) -> int:
        return topic * 15 + rng.randrange(15) if rng.random() < focus else rng.randrange(120)

    for session in range(CONFIGURATION["history_sessions"]):
        topic = session % 8
        items = [topic * 15 + (session // 8) % 15]
        items.extend(item_for(history_rng, topic) for _ in range(history_rng.randint(5, 9)))
        for position, item in enumerate(items):
            action = history_rng.choices(ACTIONS, weights=(7, 2, 1))[0]
            history.append(Event(session, item, 1_000 + session * 100 + position, action))

    for index in range(CONFIGURATION["query_sessions"]):
        topic = index % 8
        session = 10_000 + index
        query_ts = 110_000 + index * 100
        length = query_rng.randint(3, 7)
        prefix = tuple(
            Event(
                session,
                item_for(query_rng, topic),
                query_ts - length + position - 1,
                query_rng.choices(ACTIONS, weights=(8, 2, 1))[0],
            )
            for position in range(length)
        )
        queries.append(Query(session, query_ts, prefix))
        counts = {
            "clicks": target_rng.randint(1, 3),
            "carts": target_rng.randint(1, 2) if target_rng.random() < 0.8 else 0,
            "orders": 1 if target_rng.random() < 0.5 else 0,
        }
        position = 1
        for action in ACTIONS:
            for _ in range(counts[action]):
                item = (
                    target_rng.choice(prefix).item
                    if target_rng.random() < 0.35
                    else item_for(target_rng, topic, focus=0.75)
                )
                targets.append(Event(session, item, query_ts + position, action))
                position += 1
    return DemoData(seed, tuple(history), tuple(queries), tuple(targets))


def fit_history(history: Iterable[Event], cutoff: int) -> DemoModel:
    """Count symmetric, unique-item session pairs and per-action popularity.

    Reject events at or after the cutoff; silently dropping them could conceal an
    upstream split error. Repeated interactions count toward popularity, while a
    pair counts at most once per historical session.
    """
    _integer(cutoff, "cutoff")
    sessions: dict[int, set[int]] = defaultdict(set)
    popularity: dict[str, Counter[int]] = {action: Counter() for action in ACTIONS}
    for event in history:
        _event(event)
        if event.ts >= cutoff:
            raise ValueError("Historical events must occur strictly before the cutoff")
        sessions[event.session].add(event.item)
        popularity[event.action][event.item] += 1
    if not sessions:
        raise ValueError("History must contain at least one event")
    pairs: dict[int, Counter[int]] = defaultdict(Counter)
    catalog: set[int] = set()
    for items in sessions.values():
        catalog.update(items)
        for source in sorted(items):
            pairs[source].update(target for target in sorted(items) if target != source)
    return DemoModel(
        cutoff=cutoff,
        catalog=tuple(sorted(catalog)),
        popularity={action: dict(counts) for action, counts in popularity.items()},
        cooccurrence={item: dict(counts) for item, counts in pairs.items()},
    )


def predict(
    model: DemoModel, queries: Iterable[Query], k: int = 20, candidate_limit: int = 50
) -> dict[str, Any]:
    """Rank a bounded pool using only history and each observed query prefix.

    Each item receives 0.6 * normalized co-occurrence + 0.3 * reciprocal prefix
    recency + 0.1 * normalized action popularity. Co-occurrence sums over prefix
    items with the same reciprocal-recency weights. All ties use item ID ascending.
    Previously observed items remain eligible because repeat intent is legitimate.
    """
    if not 1 <= _integer(k, "k") <= 20 or _integer(candidate_limit, "candidate_limit") < k:
        raise ValueError("Require 1 <= k <= 20 and candidate_limit >= k")
    if tuple(sorted(set(model.catalog))) != model.catalog:
        raise ValueError("Model catalog must contain unique ascending item IDs")
    seen: set[int] = set()
    rows = []
    for query in queries:
        _integer(query.session, "query session")
        _integer(query.query_ts, "query timestamp")
        if query.session in seen:
            raise ValueError("Query sessions must be unique")
        seen.add(query.session)
        if query.query_ts < model.cutoff:
            raise ValueError("Query cannot precede the historical cutoff")
        for event in query.prefix:
            _event(event)
            if event.session != query.session or event.ts >= query.query_ts:
                raise ValueError("Prefix events must belong to the query and precede its timestamp")
        prefix = sorted(query.prefix, key=lambda event: (event.ts, event.item, event.action))
        recency: dict[int, float] = {}
        for distance, event in enumerate(reversed(prefix), start=1):
            recency[event.item] = max(recency.get(event.item, 0.0), 1.0 / distance)
        association: dict[int, float] = defaultdict(float)
        for source, weight in sorted(recency.items()):
            for target, count in sorted(model.cooccurrence.get(source, {}).items()):
                association[target] += weight * count
        maximum = max(association.values(), default=0.0) or 1.0
        objectives = {}
        for action in ACTIONS:
            popularity = model.popularity.get(action, {})
            popular_max = max(popularity.values(), default=0) or 1
            components = {
                item: {
                    "cooccurrence": association.get(item, 0.0) / maximum,
                    "prefix_recency": recency.get(item, 0.0),
                    "popularity": popularity.get(item, 0) / popular_max,
                }
                for item in model.catalog
            }
            scores = {
                item: 0.6 * values["cooccurrence"]
                + 0.3 * values["prefix_recency"]
                + 0.1 * values["popularity"]
                for item, values in components.items()
            }
            candidates = sorted(model.catalog, key=lambda item: (-scores[item], item))[
                :candidate_limit
            ]
            recommendations = candidates[:k]
            source_names = {
                "cooccurrence": "session co-occurrence",
                "prefix_recency": "observed prefix",
                "popularity": "historical popularity",
            }
            objectives[action] = {
                "candidates": candidates,
                "recommendations": recommendations,
                "explanations": [
                    {
                        "item": item,
                        "score": scores[item],
                        "sources": [
                            name for key, name in source_names.items() if components[item][key] > 0
                        ],
                        "features": components[item],
                    }
                    for item in recommendations
                ],
            }
        rows.append(
            {"session": query.session, "query_ts": query.query_ts, "objectives": objectives}
        )
    return {
        "schema_version": 1,
        "k": k,
        "candidate_limit": candidate_limit,
        "rows": sorted(rows, key=lambda row: row["session"]),
    }


def _write_once(path: Path, content: bytes) -> None:
    """Create an artifact exclusively, or accept byte-identical existing content."""
    if path.is_symlink():
        raise ValueError(f"Refusing a symlink artifact: {path.name}")
    try:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite existing artifact: {path.name}") from exc


def seal_predictions(predictions: Mapping[str, Any], path: str | Path) -> str:
    """Persist immutable prediction bytes before labels are supplied to evaluation."""
    content = canonical_json(predictions)
    _write_once(Path(path), content)
    return _sha(content)


def _prediction_rows(payload: Any) -> dict[int, dict[str, Any]]:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported prediction schema")
    k = _integer(payload.get("k"), "k")
    limit = _integer(payload.get("candidate_limit"), "candidate_limit")
    if not 1 <= k <= 20 or limit < k or not isinstance(payload.get("rows"), list):
        raise ValueError("Invalid prediction limits or rows")
    rows = {}
    for row in payload["rows"]:
        session = _integer(row["session"], "session")
        _integer(row["query_ts"], "query timestamp")
        if session in rows or set(row["objectives"]) != set(ACTIONS):
            raise ValueError("Duplicate session or incomplete objective coverage")
        for action in ACTIONS:
            objective = row["objectives"][action]
            candidates, recommendations = objective["candidates"], objective["recommendations"]
            for items, maximum in ((candidates, limit), (recommendations, k)):
                if not isinstance(items, list) or len(items) > maximum:
                    raise ValueError("Invalid recommendation or candidate length")
                for item in items:
                    _integer(item, "item")
                if len(set(items)) != len(items):
                    raise ValueError("Duplicate recommendation or candidate item")
            if not set(recommendations).issubset(candidates):
                raise ValueError("Recommendations must come from the candidate pool")
        rows[session] = row
    return rows


def evaluate_predictions(
    prediction_path: str | Path, targets: Iterable[Event], expected_sha256: str
) -> dict[str, Any]:
    """Evaluate sealed predictions with pooled, deduplicated Recall@20 counts.

    The denominator for each objective is the sum of min(20, distinct target
    items) across sessions. The candidate oracle counts the maximum hits possible
    from each saved candidate pool, capped by that same per-session denominator.
    Sessions without targets for an objective contribute zero to its denominator.
    """
    path = Path(prediction_path)
    if path.is_symlink():
        raise ValueError("Refusing a symlink prediction file")
    content = path.read_bytes()
    if _sha(content) != expected_sha256:
        raise ValueError("Prediction checksum mismatch")
    rows = _prediction_rows(json.loads(content))
    truth: dict[tuple[int, str], set[int]] = defaultdict(set)
    for target in targets:
        _event(target)
        if target.session not in rows:
            raise ValueError("Target session has no prediction")
        if target.ts <= rows[target.session]["query_ts"]:
            raise ValueError("Targets must occur strictly after the query timestamp")
        truth[target.session, target.action].add(target.item)
    objectives = {}
    for action in ACTIONS:
        hits = denominator = candidate_hits = 0
        for session, row in rows.items():
            relevant = truth[session, action]
            denominator += min(20, len(relevant))
            objective = row["objectives"][action]
            hits += len(relevant.intersection(objective["recommendations"]))
            candidate_hits += min(20, len(relevant.intersection(objective["candidates"])))
        objectives[action] = {
            "weight": WEIGHTS[action],
            "hits": hits,
            "denominator": denominator,
            "recall_at_20": hits / denominator if denominator else 0.0,
            "candidate_hits": candidate_hits,
            "candidate_oracle": candidate_hits / denominator if denominator else 0.0,
        }
    return {
        "weighted_recall_at_20": sum(
            values["weight"] * values["recall_at_20"] for values in objectives.values()
        ),
        "weighted_candidate_oracle": sum(
            values["weight"] * values["candidate_oracle"] for values in objectives.values()
        ),
        "objectives": objectives,
    }


def _manifest_base(seed: int) -> dict[str, Any]:
    return {
        "owner": _OWNER,
        "schema_version": 1,
        "seed": seed,
        "configuration": CONFIGURATION,
        "implementation_sha256": _sha(Path(__file__).read_bytes()),
        "renderer_sha256": _sha(Path(__file__).with_name("public_demo_report.py").read_bytes()),
    }


def _read_manifest(output: Path) -> dict[str, Any]:
    path = output / "manifest.json"
    if output.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError("Output directory lacks a regular owned manifest")
    manifest = json.loads(path.read_bytes())
    if not isinstance(manifest, dict) or manifest.get("owner") != _OWNER:
        raise ValueError("Output directory is not owned by this demo")
    allowed = {*_ARTIFACTS, "manifest.json"}
    if manifest.get("status") == "started":
        allowed.add("manifest.complete.tmp")
    unknown = {path.name for path in output.iterdir()} - allowed
    if unknown:
        raise ValueError(f"Output directory contains unknown entries: {sorted(unknown)}")
    return manifest


def verify_output(output_dir: str | Path, seed: int | None = None) -> dict[str, Any]:
    """Verify every completed artifact without fitting or rewriting any file."""
    output = Path(output_dir)
    manifest = _read_manifest(output)
    actual_seed = _integer(manifest.get("seed"), "manifest seed")
    if seed is not None and seed != actual_seed:
        raise ValueError("Existing output was generated with a different seed")
    for key, value in _manifest_base(actual_seed).items():
        if manifest.get(key) != value:
            raise ValueError(f"Manifest contract mismatch: {key}")
    if manifest.get("status") != "completed" or set(manifest.get("files", {})) != set(_ARTIFACTS):
        raise ValueError("Output manifest is incomplete")
    for name in _ARTIFACTS:
        path = output / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or unsafe artifact: {name}")
        content = path.read_bytes()
        if manifest["files"][name] != {"sha256": _sha(content), "bytes": len(content)}:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    result: dict[str, Any] = json.loads((output / "result.json").read_bytes())
    prediction_sha = manifest["files"]["predictions.json"]["sha256"]
    if result["provenance"]["predictions_sha256"] != prediction_sha:
        raise ValueError("Result and prediction seal disagree")
    return result


def run_demo(output_dir: str | Path, seed: int = 42) -> dict[str, Any]:
    """Execute or verify a deterministic demo in a new or previously owned directory.

    Unknown files and mismatched artifacts are never overwritten. An interrupted
    owned run may resume when every existing artifact matches its regenerated
    bytes. A completed run verifies checksums and returns its saved result.
    """
    _integer(seed, "seed")
    output = Path(output_dir)
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("Output must be a regular directory")
    output.mkdir(parents=True, exist_ok=True)
    base = _manifest_base(seed)
    started = {**base, "status": "started", "files": {}}
    if any(output.iterdir()):
        manifest = _read_manifest(output)
        if manifest.get("status") == "completed":
            return verify_output(output, seed)
        if manifest != started:
            raise ValueError("Incomplete output has an incompatible manifest")
    else:
        _write_once(output / "manifest.json", canonical_json(started))

    data = generate_synthetic(seed)
    model = fit_history(data.history, HISTORY_CUTOFF)
    predictions = predict(model, data.queries)
    predictions_sha = seal_predictions(predictions, output / "predictions.json")
    metrics = evaluate_predictions(output / "predictions.json", data.targets, predictions_sha)
    row_lookup = {row["session"]: row for row in predictions["rows"]}
    examples = []
    for query in data.queries[:3]:
        objectives = row_lookup[query.session]["objectives"]
        examples.append(
            {
                "session": query.session,
                "query_ts": query.query_ts,
                "prefix": [
                    {"item": event.item, "action": event.action, "ts": event.ts}
                    for event in query.prefix
                ],
                "targets": {
                    action: sorted(
                        {
                            event.item
                            for event in data.targets
                            if event.session == query.session and event.action == action
                        }
                    )
                    for action in ACTIONS
                },
                **{
                    field: {action: objectives[action][field] for action in ACTIONS}
                    for field in ("recommendations", "candidates", "explanations")
                },
            }
        )
    result = {
        "schema_version": 1,
        "demo": "Synthetic session recommendation",
        "seed": seed,
        "synthetic": True,
        "scope": (
            "Small deterministic software demonstration; not a production or competition benchmark."
        ),
        "configuration": dict(CONFIGURATION),
        "data_summary": {
            "history_events": len(data.history),
            "observed_events": sum(len(query.prefix) for query in data.queries),
            "target_events": len(data.targets),
            "history_cutoff": HISTORY_CUTOFF,
            "earliest_query_ts": min(query.query_ts for query in data.queries),
        },
        "metrics": metrics,
        "examples": examples,
        "provenance": {
            "history_sha256": _sha(canonical_json([asdict(event) for event in data.history])),
            "queries_sha256": _sha(canonical_json([asdict(query) for query in data.queries])),
            "predictions_sha256": predictions_sha,
            "predictions_sealed_before_evaluation": True,
            "targets_used_in_fit_or_inference": False,
            "tie_break": "score descending, item ID ascending",
        },
        "stages": [
            {"name": name, "status": "completed"}
            for name in ("generate", "fit", "predict", "seal", "evaluate")
        ],
    }
    from otto_recsys.public_demo_report import render_report

    _write_once(output / "result.json", canonical_json(result))
    _write_once(output / "report.html", render_report(result).encode("utf-8"))
    files = {}
    for name in _ARTIFACTS:
        content = (output / name).read_bytes()
        files[name] = {"sha256": _sha(content), "bytes": len(content)}
    completed = {**base, "status": "completed", "files": files}
    # Replace only the exact marker created by this run, never an unknown file.
    if (output / "manifest.json").read_bytes() != canonical_json(started):
        raise ValueError("Manifest changed during execution")
    temporary = output / "manifest.complete.tmp"
    _write_once(temporary, canonical_json(completed))
    os.replace(temporary, output / "manifest.json")
    return verify_output(output, seed)
