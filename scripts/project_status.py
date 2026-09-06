from __future__ import annotations

import json
import subprocess
from pathlib import Path


def command(*args: str) -> str:
    completed = subprocess.run(args, check=False, capture_output=True, text=True)
    return completed.stdout.strip()


def load(path: str) -> dict[str, object] | None:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


print("=" * 72)
print("OTTO PROJECT STATUS")
print("=" * 72)
print(f"branch={command('git', 'branch', '--show-current')}")
print(f"commit={command('git', 'rev-parse', '--short', 'HEAD')}")
status = command("git", "status", "--short")
print(f"working_tree={'clean' if not status else 'modified'}")

ranking = load("artifacts/ranking_training_cache/manifest.json")
negative = load("artifacts/hard_negatives/manifest.json")
items = load("artifacts/two_tower_inputs/items/manifest.json")

if ranking:
    print(f"ranking_sessions={ranking.get('sessions')}")
    print(f"ranking_labels={ranking.get('label_rows')}")

if negative:
    print(f"hard_negative_buckets={negative.get('completed_buckets')}/32")
    print(f"hard_negative_rows={negative.get('output_rows')}")
    print(f"hard_negative_family_sha256={negative.get('family_sha256')}")

if items:
    print(f"two_tower_items={items.get('items')}")
    print(f"two_tower_dimension={items.get('dimension')}")


pipeline = load("artifacts/two_tower_pipeline/latest.json")
if pipeline:
    print(f"two_tower_pipeline={pipeline.get('pipeline_name')}")
    print(f"two_tower_pipeline_run_id={pipeline.get('run_id')}")

resume_proof = load("reports/metrics/two_tower_resume_proof.json")
if resume_proof:
    print(f"two_tower_resume_proof={resume_proof.get('status')}")
    print(f"two_tower_resumed_from_step={resume_proof.get('resumed_from_step')}")
    print(f"two_tower_resume_final_step={resume_proof.get('final_step')}")

fold0 = load("reports/metrics/two_tower_fold0_training.json")
if fold0:
    print(f"two_tower_fold0_training={fold0.get('status')}")
    print(f"two_tower_fold0_global_step={fold0.get('global_step')}")
    print(f"two_tower_fold0_best_valid_loss={fold0.get('best_valid_loss')}")
else:
    print("two_tower_fold0_training=pending")

print("OTTO_PROJECT_STATUS_COMPLETE")
