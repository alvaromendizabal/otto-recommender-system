"""Apply versioned sections before executing and publishing canonical notebooks.

This changes source cells only. The existing publication gate must execute twice
and verify outputs and receipts before any notebook is committed.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "notebooks"
    for path in sorted((root / "sections").glob("*.json")):
        section = json.loads(path.read_text())
        notebook_path = root / section["notebook"]
        if notebook_path.parent != root or not notebook_path.name.endswith(".ipynb"):
            raise ValueError("section must target a canonical notebook")
        notebook = json.loads(notebook_path.read_text())
        ids = [cell["id"] for cell in section["cells"]]
        if len(set(ids)) != len(ids):
            raise ValueError("section contains duplicate cell IDs")
        existing = {cell.get("id"): cell for cell in notebook["cells"]}
        present = [identity in existing for identity in ids]
        if any(present):
            if not all(present):
                raise ValueError("section is only partially present")
            for cell in section["cells"]:
                prior = existing[cell["id"]]
                if prior["cell_type"] != cell["cell_type"] or "".join(prior["source"]) != "".join(
                    cell["source"]
                ):
                    raise ValueError("published section differs from its source declaration")
            continue
        notebook["cells"].extend(section["cells"])
        notebook_path.write_text(json.dumps(notebook, indent=1) + "\n")
        print(f"Prepared {len(ids)} source cells for {notebook_path.name}")


if __name__ == "__main__":
    main()
