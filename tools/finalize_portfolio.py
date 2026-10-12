"""Add idempotent navigation to existing documentation without replacing it."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSERTS = {
    "REPRODUCIBILITY.md": "**Start with the [clone-to-results guide](QUICKSTART.md).** The new reproduce.py entry point supports dependency-free review, genuine native training on invented records, and a reader-supplied raw-data reference run. The detailed stage-level commands below remain available. Later neural-release exact reconstruction is a separate, incomplete scope.",
    "PUBLICATION_SCOPE.md": "## Clone-to-results scope\n\nThe [public launcher](../reproduce.py) now connects the actual raw converter, historical retrieval, feature engine, matched LightGBM fitting, sealed temporal evaluation and CSV validation. Its small profile trains on invented records; full reference research accepts independently obtained official training JSONL. Neither profile claims to recreate the later private neural-enhanced release or its exact score. No owner cloud credentials are required.\n\nHistorical public deployment-account numbers have been replaced with reserved placeholders. [Redaction provenance](../reports/privacy/redactions.json) separates original file hashes from current public bytes. Model weights and measured scores are not changed by this redaction. Git history has not been rewritten. Never reuse a placeholder ARN or bucket as a live deployment target.",
    "REVIEWER_GUIDE.md": "**Fastest review:** read the [engineering decisions](DECISIONS.md), run the [native reference walkthrough](QUICKSTART.md), then inspect its sealed evaluation and native model files. The [current closeout](RESEARCH_STATUS.md) supersedes archived stage-status narratives.",
    "EMPLOYER_OVERVIEW.md": "**Release complete; reproducible public reference available.** A single documented entry point now connects raw JSONL to actual native training, temporal evaluation and verified recommendations. [Clone-to-results](QUICKSTART.md) · [Engineering decisions](DECISIONS.md) · [Final research status](RESEARCH_STATUS.md).",
}

if __name__ == "__main__":
    for name, paragraph in INSERTS.items():
        path = ROOT / "docs" / name
        text = path.read_text()
        if "QUICKSTART.md" in text:
            continue
        heading, rest = text.split("\n", 1)
        path.write_text(heading + "\n\n" + paragraph + "\n" + rest)
