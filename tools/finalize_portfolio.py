"""Finalize public source and add navigation without replacing detailed documentation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSERTS = {
    "REPRODUCIBILITY.md": "**Start with the [clone-to-results guide](QUICKSTART.md).** The new reproduce.py entry point supports dependency-free review, genuine native training on invented records, and a reader-supplied raw-data reference run. Detailed stage-level commands below remain available. Later neural-release exact reconstruction is a separate, incomplete scope.",
    "PUBLICATION_SCOPE.md": "## Clone-to-results scope\n\nThe [public launcher](../reproduce.py) connects raw conversion, historical retrieval, features, matched LightGBM fitting, sealed temporal evaluation and CSV validation. The small profile uses invented records; full reference research accepts independently obtained official training JSONL. Neither promises exact reconstruction of the later private neural release. No owner cloud credentials are required.\n\nPublic deployment-account numbers are replaced with reserved placeholders. [Redaction provenance](../reports/privacy/redactions.json) distinguishes original hashes from current public bytes. Measured scores and model weights are unchanged. Git history is not rewritten; placeholder cloud locations are not live deployment targets.",
    "REVIEWER_GUIDE.md": "**Fastest review:** [engineering decisions](DECISIONS.md), [native reference walkthrough](QUICKSTART.md), then sealed evaluation and native model files. The [current closeout](RESEARCH_STATUS.md) supersedes archived stage-status narratives.",
    "EMPLOYER_OVERVIEW.md": "**Release complete; reproducible public reference available.** One documented entry point connects raw JSONL to native training, temporal evaluation and verified recommendations. [Clone-to-results](QUICKSTART.md) · [Engineering decisions](DECISIONS.md) · [Final research status](RESEARCH_STATUS.md).",
}

if __name__ == "__main__":
    for name, paragraph in INSERTS.items():
        path = ROOT / "docs" / name
        text = path.read_text()
        if "QUICKSTART.md" in text:
            continue
        heading, rest = text.split("\n", 1)
        path.write_text(heading + "\n\n" + paragraph + "\n" + rest)
    path = ROOT / "reproduce.py"
    text = path.read_text()
    if "from contextlib import suppress" not in text:
        text = text.replace("import argparse\n", "import argparse\nfrom contextlib import suppress\n")
    for spaces, sig in ((12, "SIGTERM"), (16, "SIGKILL")):
        pad = " " * spaces
        before = f"{pad}try:\n{pad}    os.killpg(process.pid, signal.{sig})\n{pad}except ProcessLookupError:\n{pad}    pass"
        after = f"{pad}with suppress(ProcessLookupError):\n{pad}    os.killpg(process.pid, signal.{sig})"
        text = text.replace(before, after)
    path.write_text(text)
    path = ROOT / "scripts/reproduce_reference.py"
    text = path.read_text().replace("predeclared small integration profile; not the historical 102-feature model", "Small fixture profile; not the historical model")
    text = text.replace("Use the locked project environment for train/research; completed outputs are preserved.", "Use the locked ML environment. Completed outputs are preserved.")
    path.write_text(text)
