"""Publish compact, audited research evidence for notebook replay and review."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import polars as pl

from otto_recsys.experiments.manifest import sha256_file
from otto_recsys.research.evaluation import verify_seal
from otto_recsys.research.features import feature_catalog
from otto_recsys.research.protocol import atomic_json


def read(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text()))


def table(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def publish(root: Path, destination: Path) -> dict[str, Any]:
    audit = read(root / "audit/report.json")
    seal = verify_seal(root)
    interpretation = read(root / "interpretation/report.json")
    if (
        audit["status"] != "passed"
        or audit["seal_id"] != seal["seal_id"]
        or audit["evaluation_report_sha256"] != sha256_file(root / "evaluation/report.json")
        or interpretation["status"] != "passed"
        or interpretation["seal_id"] != seal["seal_id"]
    ):
        raise ValueError("publication requires matching successful audit and explanation")
    for name, digest in interpretation["files"].items():
        if sha256_file(root / "interpretation" / name) != digest:
            raise ValueError("interpretation checksum mismatch")
    for name, digest in audit["ablations"]["native_model_sha256"].items():
        if sha256_file(root / name) != digest:
            raise ValueError("audited model changed before publication")
    destination.mkdir(parents=True, exist_ok=True)
    copies = {
        "evaluation.json": "evaluation/report.json",
        "ablations.json": "models/ablation_report.json",
        "audit.json": "audit/report.json",
        "interpretation.json": "interpretation/report.json",
        "evaluation_seal.json": "evaluation_seal.json",
        "temporal_manifest.json": "corpus/manifest.json",
        "temporal_contract.json": "corpus/contract.json",
        "screening.json": "screening/selection.json",
        "managed_study.json": "job_status.json",
    }
    for target, source in copies.items():
        shutil.copyfile(root / source, destination / target)
    definitions = {f.name: f for f in feature_catalog()}
    screening = read(root / "screening/selection.json")
    chosen = {n for model in seal["models"].values() for n in model["features"]}
    if set(definitions) != {r["name"] for r in screening["features"]}:
        raise ValueError("published catalog differs from the screened feature definitions")
    features = []
    for row in screening["features"]:
        definition = definitions[row["name"]]
        feature = {**row}
        correlations = feature.pop("target_correlations", {})
        feature.update(
            definition=definition.definition,
            availability=definition.availability,
            final_selected=row["name"] in chosen,
            final_reason="selected"
            if row["name"] in chosen
            else ("source_family_ablation" if row["status"] == "retained" else row["status"]),
            **{f"correlation_{o}": correlations.get(o) for o in ("clicks", "carts", "orders")},
        )
        features.append(feature)
    # Quality-rejected features have no pilot importance; retain an explicit blank.
    keys = list(dict.fromkeys(k for row in features for k in row))
    table(destination / "feature_catalog.csv", [{k: row.get(k) for k in keys} for row in features])
    importance = pl.read_parquet(root / "interpretation/feature_importance.parquet")
    importance.write_csv(destination / "feature_importance.csv")
    ablation = read(root / "models/ablation_report.json")
    variants = []
    for name, row in ablation["variants"].items():
        for objective, model in row["models"].items():
            variants.append(
                {
                    "variant": name,
                    "objective": objective,
                    "features": row["features"],
                    **row["objectives"][objective],
                    **model,
                    "weighted_recall_at_20": row["weighted_recall_at_20"],
                    "chosen": ablation["chosen"][objective] == name,
                }
            )
    table(destination / "ablation_models.csv", variants)
    counts = {
        family: {
            "engineered": sum(f.family == family for f in definitions.values()),
            "screened": screening["retained_families"].get(family, 0),
            "final": sum(definitions[n].family == family for n in chosen),
        }
        for family in sorted({f.family for f in definitions.values()})
    }
    table(destination / "feature_families.csv", [{"family": k, **v} for k, v in counts.items()])
    files = {
        p.name: sha256_file(p)
        for p in sorted(destination.iterdir())
        if p.is_file() and p.name != "manifest.json"
    }
    manifest = {
        "status": "passed",
        "seal_id": seal["seal_id"],
        "audit_id": audit["audit_id"],
        "features": {
            "engineered": len(definitions),
            "screened": screening["retained_count"],
            "final": len(chosen),
            "families": counts,
        },
        "files": files,
        "scope": "audited temporal study and post-selection diagnostics; no Kaggle score",
    }
    atomic_json(destination / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/research"))
    parser.add_argument("--destination", type=Path, default=Path("reports/research"))
    args = parser.parse_args()
    print(json.dumps(publish(args.root, args.destination), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
