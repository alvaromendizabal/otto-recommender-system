"""Independently reconcile the complete synthetic recommendation evaluation.

Reads published JSON boundaries using only Python's standard library. Does not
import the retriever, ranker, target generator or project metric implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from html import escape
from pathlib import Path
from typing import Any

ACTIONS = ("clicks", "carts", "orders")
WEIGHTS = {"clicks": 0.1, "carts": 0.3, "orders": 0.6}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def integer(value: Any, name: str) -> int:
    require(type(value) is int and value >= 0, f"Invalid {name}")
    return value


def unique_items(items: Any, maximum: int) -> set[int]:
    require(isinstance(items, list) and len(items) <= maximum, "Invalid candidate/ranking length")
    values = {integer(item, "item identifier") for item in items}
    require(len(values) == len(items), "Duplicate candidate/ranking item")
    return values


def audit_payload(predictions: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    require(predictions.get("schema_version") == 1, "Unknown prediction schema")
    require(evaluation.get("schema_version") == 1, "Unknown evaluation schema")
    require(evaluation.get("synthetic") is True, "Only synthetic evaluation is supported")
    require(integer(predictions.get("k"), "k") == 20, "This audit requires top-20 predictions")
    limit = integer(predictions.get("candidate_limit"), "candidate limit")
    require(limit >= 20, "Candidate pool must allow top-20 rankings")
    rows = {}
    require(isinstance(predictions.get("rows"), list), "Missing prediction population")
    for row in predictions["rows"]:
        session = integer(row["session"], "session identifier")
        require(session not in rows, "Duplicate prediction session")
        integer(row["query_ts"], "query time")
        require(set(row["objectives"]) == set(ACTIONS), "Incomplete objective coverage")
        for objective in row["objectives"].values():
            candidates = unique_items(objective["candidates"], limit)
            ranked = unique_items(objective["recommendations"], 20)
            require(ranked <= candidates, "Ranked item is absent from candidate pool")
        rows[session] = row
    require(bool(rows), "Empty prediction population")
    queries = {}
    require(isinstance(evaluation.get("queries"), list), "Missing query population")
    for query in evaluation["queries"]:
        session = integer(query["session"], "query session")
        require(session not in queries and session in rows, "Invalid query coverage")
        integer(query["query_ts"], "query time")
        integer(query["prefix_events"], "prefix length")
        require(query["query_ts"] == rows[session]["query_ts"], "Query time mismatch")
        queries[session] = query
    require(set(queries) == set(rows), "Query/prediction population mismatch")
    targets: dict[tuple[int, str], set[int]] = {}
    require(isinstance(evaluation.get("targets"), list), "Missing targets")
    for target in evaluation["targets"]:
        session = integer(target["session"], "target session")
        item = integer(target["item"], "target item")
        timestamp = integer(target["ts"], "target time")
        action = target["action"]
        require(session in rows, "Target session has no prediction")
        require(action in ACTIONS, "Unknown target objective")
        require(timestamp > queries[session]["query_ts"], "Target is not strictly in the future")
        targets.setdefault((session, action), set()).add(item)

    def summarize(sessions: list[int]) -> dict[str, Any]:
        objectives = {}
        for action in ACTIONS:
            hits = reachable = denominator = active = 0
            for session in sessions:
                truth = targets.get((session, action), set())
                objective = rows[session]["objectives"][action]
                hits += len(truth.intersection(objective["recommendations"]))
                reachable += min(20, len(truth.intersection(objective["candidates"])))
                denominator += min(20, len(truth))
                active += bool(truth)
            require(0 <= hits <= reachable <= denominator, "Metric count reconciliation failed")
            objectives[action] = {
                "sessions_with_targets": active,
                "denominator": denominator,
                "hits": hits,
                "candidate_hits": reachable,
                "ranking_shortfall": reachable - hits,
                "retrieval_shortfall": denominator - reachable,
                "recall_at_20": hits / denominator if denominator else 0.0,
                "candidate_oracle": reachable / denominator if denominator else 0.0,
                "ranking_shortfall_rate": (reachable - hits) / denominator if denominator else 0.0,
                "retrieval_shortfall_rate": (
                    (denominator - reachable) / denominator if denominator else 0.0
                ),
            }
        return {
            "sessions": len(sessions),
            "objectives": objectives,
            "weighted_recall_at_20": math.fsum(
                WEIGHTS[action] * row["recall_at_20"] for action, row in objectives.items()
            ),
            "weighted_candidate_oracle": math.fsum(
                WEIGHTS[action] * row["candidate_oracle"] for action, row in objectives.items()
            ),
            "weighted_ranking_shortfall": math.fsum(
                WEIGHTS[action] * row["ranking_shortfall_rate"]
                for action, row in objectives.items()
            ),
            "weighted_retrieval_shortfall": math.fsum(
                WEIGHTS[action] * row["retrieval_shortfall_rate"]
                for action, row in objectives.items()
            ),
            "weighted_evaluable_mass": math.fsum(
                WEIGHTS[action] for action, row in objectives.items() if row["denominator"]
            ),
        }

    sessions = sorted(rows)
    return {
        "schema_version": 1,
        "status": "passed",
        "synthetic": True,
        "scope": "Independent audit of the complete invented evaluation; no historical score claim",
        "metric": "Per-objective pooled hits / sum(min(20, distinct targets per session))",
        "count_identity": "hits + ranking_shortfall + retrieval_shortfall = denominator",
        "weights": WEIGHTS,
        "overall": summarize(sessions),
        "prefix_slices": {
            "0_to_4_events": summarize([s for s in sessions if queries[s]["prefix_events"] <= 4]),
            "5_or_more_events": summarize(
                [s for s in sessions if queries[s]["prefix_events"] >= 5]
            ),
        },
        "limitations": [
            "Candidate shortfalls are theoretical headroom, not a promised achievable gain.",
            "Zero-target objectives have a zero denominator and contribute zero weighted mass.",
            "Synthetic sessions do not establish offline generalization or online business lift.",
        ],
    }


def audit(directory: Path) -> dict[str, Any]:
    require(directory.is_dir() and not directory.is_symlink(), "Invalid demo directory")
    expected = {"predictions.json", "evaluation.json", "result.json", "report.html"}
    manifest_path = directory / "manifest.json"
    require(manifest_path.is_file() and not manifest_path.is_symlink(), "Missing regular manifest")
    manifest = json.loads(manifest_path.read_bytes())
    require(manifest.get("status") == "completed", "Incomplete generating run")
    require(set(manifest.get("files", {})) == expected, "Unexpected artifact manifest")
    require(
        {p.name for p in directory.iterdir()} == expected | {"manifest.json"}, "Extra artifacts"
    )
    hashes = {}
    for name in sorted(expected):
        path = directory / name
        require(path.is_file() and not path.is_symlink(), "Invalid source artifact")
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        require(
            manifest["files"][name] == {"sha256": digest, "bytes": len(content)}, "Changed source"
        )
        hashes[name] = digest
    predictions = json.loads((directory / "predictions.json").read_bytes())
    evaluation = json.loads((directory / "evaluation.json").read_bytes())
    recorded = json.loads((directory / "result.json").read_bytes())
    require(recorded.get("synthetic") is True, "Result is not synthetic")
    require(
        evaluation["predictions_sha256"] == hashes["predictions.json"], "Prediction seal mismatch"
    )
    report = audit_payload(predictions, evaluation)
    metrics = recorded["metrics"]
    for action, values in report["overall"]["objectives"].items():
        for count in ("hits", "candidate_hits", "denominator"):
            require(
                values[count] == metrics["objectives"][action][count], "Recorded count mismatch"
            )
        for metric in ("recall_at_20", "candidate_oracle"):
            require(
                math.isclose(values[metric], metrics["objectives"][action][metric], abs_tol=1e-12),
                "Recorded objective metric mismatch",
            )
    for metric in ("weighted_recall_at_20", "weighted_candidate_oracle"):
        require(
            math.isclose(report["overall"][metric], metrics[metric], abs_tol=1e-12),
            "Recorded pooled metric mismatch",
        )
    hashes["manifest.json"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    report["source_sha256"] = hashes
    return report


def render_report(report: dict[str, Any]) -> str:
    overall = report["overall"]
    rows = []
    for scope, summary in {"All sessions": overall, **report["prefix_slices"]}.items():
        for action, values in summary["objectives"].items():
            rows.append(
                f"<tr><th>{escape(scope.replace('_', ' '))}</th><td>{action}</td>"
                f"<td>{summary['sessions']}</td><td>{values['sessions_with_targets']}</td>"
                f"<td>{values['hits']}</td><td>{values['ranking_shortfall']}</td>"
                f"<td>{values['retrieval_shortfall']}</td><td>{values['denominator']}</td>"
                f"<td>{values['recall_at_20']:.4f}</td></tr>"
            )
    cards = "".join(
        f'<article><p class="eyebrow">{title}</p><strong>{overall[key]:.4f}</strong>'
        f"<p>{description}</p></article>"
        for title, key, description in (
            ("Achieved", "weighted_recall_at_20", "Weighted Recall@20 on the complete fixture."),
            (
                "Inside candidate pools",
                "weighted_ranking_shortfall",
                "Reachable but absent from top-20.",
            ),
            (
                "Outside candidate pools",
                "weighted_retrieval_shortfall",
                "Unreachable with the saved pools.",
            ),
        )
    )
    return (
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Session Lab · Recommendation diagnostics</title><style>
:root{color-scheme:light;font-family:Inter,system-ui,sans-serif;color:#1e293e;background:#f5f6fb}
*{box-sizing:border-box}body{margin:0}main{max-width:1120px;margin:auto;padding:48px 24px}
header{border-top:5px solid #6250b2;padding:28px 0}h1{font-size:clamp(32px,5vw,52px);margin:12px 0}
h2{font-size:24px;margin:36px 0 12px}p{line-height:1.65;max-width:88ch}.eyebrow{font-size:12px;
letter-spacing:.07em;text-transform:uppercase;font-weight:750;color:#6250b2}.cards{display:grid;
grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:18px}article{background:#fff;
padding:24px;border:1px solid #dce0ed;border-radius:12px}strong{font-size:38px}.table{overflow:auto;
background:#fff;border:1px solid #dce0ed;border-radius:10px}
table{border-collapse:collapse;width:100%;
font-size:14px;white-space:nowrap}th,td{text-align:left;padding:13px 15px;border-bottom:1px solid
#e7eaf2}thead{background:#eeebf8}td{font-variant-numeric:tabular-nums}th{font-weight:650}
footer{border-top:1px solid #dce0ed;margin-top:36px;padding-top:16px;font-size:13px;color:#59627c}
code{font-size:13px}li{line-height:1.7}</style></head><body><main>
<header><p class="eyebrow">Session Lab / Independent evaluation</p>
<h1>Find where the relevant items went.</h1><p>This report reads sealed predictions and the
complete <b>synthetic</b> target population. It recomputes the recommendation metric without
importing the model or the original evaluator. Historical OTTO results are separate.</p></header>
<section class="cards">"""
        + cards
        + """</section>
<h2>Separate retrieval from ranking.</h2><p>Every denominator unit is accounted for as a hit,
ranking shortfall, or retrieval shortfall. Candidate coverage describes possible hits if the
pool were ranked perfectly; the difference is not a promised model improvement.</p>
<p><b>Weights:</b> clicks 10% · carts 30% · orders 60%. These three cards sum to the
evaluable objective weight. An objective with no future targets contributes zero weight.</p>
<h2>Inspect counts before interpreting rates.</h2><p>Counts pool distinct targets per session,
capped at 20. The two observed-prefix slices partition every query, including sessions with
no targets for an objective. Slice recall is recomputed from counts, never averaged.</p>
<div class="table"><table><thead><tr><th>Population</th><th>Objective</th><th>Queries</th>
<th>With targets</th><th>Hits</th><th>Ranking shortfall</th><th>Retrieval shortfall</th>
<th>Denominator</th><th>Recall@20</th></tr></thead><tbody>"""
        + "".join(rows)
        + """</tbody></table></div>
<footer>Machine-readable output records exact counts and source SHA-256 identities.
This fixture demonstrates evaluation and provenance contracts; it does not establish
generalization or online business lift.</footer></main></body></html>\n"""
    )


def write_once(path: Path, text: str) -> None:
    require(not path.is_symlink(), "Output cannot be a symlink")
    if path.exists():
        require(path.read_text() == text, "Refusing to overwrite a different audit")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--html", type=Path)
    args = parser.parse_args()
    result = audit(args.demo_output)
    write_once(args.output, json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    if args.html is not None:
        write_once(args.html, render_report(result))
    print(
        json.dumps(
            {"status": "passed", "synthetic": True, "sessions": result["overall"]["sessions"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
