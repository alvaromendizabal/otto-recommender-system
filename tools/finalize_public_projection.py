"""One-time, bounded repair of nested public metadata digests after redaction.

No model implementation, measurements, predictions or acceptance thresholds change.
Original publication identities remain in a separate provenance receipt.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
BEFORE = "95fa2c6a3ad99f6e501fc5ce80b08502f633cc64"
PATTERN = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
SUFFIXES = {".json", ".toml", ".md", ".py", ".ipynb", ".html", ".svg", ".txt", ".yml", ".yaml", ".csv"}


def sha(value):
    return hashlib.sha256(value).hexdigest()


def fingerprint(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def parse(name, text):
    try:
        return tomllib.loads(text) if name.endswith(".toml") else json.loads(text)
    except (ValueError, TypeError):
        return None


def pairs(left, right, location):
    if type(left) is not type(right) or left == right:
        return
    if isinstance(left, dict):
        yield fingerprint(left), fingerprint(right), location
        for key in ("receipt_id", "audit_id", "seal_id", "input_id", "protocol_id", "contract_id"):
            if key in left and key in right:
                yield (fingerprint({k: v for k, v in left.items() if k != key}),
                       fingerprint({k: v for k, v in right.items() if k != key}),
                       location + "#without:" + key)
        for key in left.keys() & right.keys():
            yield from pairs(left[key], right[key], location + "/" + str(key))
    elif isinstance(left, list) and len(left) == len(right):
        yield fingerprint(left), fingerprint(right), location
        for index, (a, b) in enumerate(zip(left, right)):
            yield from pairs(a, b, location + "/" + str(index))


def main():
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BEFORE], cwd=ROOT).decode().splitlines()
    paths = [n for n in paths if Path(n).suffix in SUFFIXES and (ROOT / n).is_file()]
    original = {n: subprocess.check_output(["git", "show", BEFORE + ":" + n], cwd=ROOT).decode() for n in paths}
    base = {n: (ROOT / n).read_text() for n in paths}
    used = set().union(*(set(PATTERN.findall(text)) for text in base.values()))
    parsed_original = {n: parse(n, text) for n, text in original.items() if n.endswith((".json", ".toml"))}
    parsed_base = {n: parse(n, base[n]) for n in parsed_original}
    current = dict(base)
    for iteration in range(50):
        mapping, locations = {}, {}

        def add(before, after, location):
            if before == after or before not in used:
                return
            if before in mapping and mapping[before] != after:
                raise ValueError("Ambiguous nested public identity; no automatic change")
            mapping[before], locations[before] = after, location

        for name, text in current.items():
            add(sha(original[name].encode()), sha(text.encode()), name + "#original-bytes")
            add(sha(base[name].encode()), sha(text.encode()), name + "#prior-public-bytes")
        for name, old in parsed_original.items():
            new, prior = parse(name, current[name]), parsed_base[name]
            if old is not None and new is not None:
                for before, after, location in pairs(old, new, name):
                    add(before, after, location + "#original")
            if prior is not None and new is not None:
                for before, after, location in pairs(prior, new, name):
                    add(before, after, location + "#prior-public")
        updated = {n: PATTERN.sub(lambda m: mapping.get(m[0], m[0]), text) for n, text in base.items()}
        if updated == current:
            break
        current = updated
    else:
        raise ValueError("Public metadata graph did not converge")
    changed = {n: text for n, text in current.items() if text != base[n]}
    if changed and len(changed) != 98:
        raise ValueError("Publication tree differs from the 98-file locally validated projection")
    for name, text in changed.items():
        if PATTERN.sub("<digest>", text) != PATTERN.sub("<digest>", base[name]):
            raise ValueError("Non-digest mutation refused")
        if not name.startswith(("configs/", "reports/", "research/manual/otto_feature_round01/")):
            raise ValueError("Change outside the metadata dependency inventory")
    receipt_path = ROOT / "reports/privacy/redactions.json"
    receipt = json.loads(receipt_path.read_text())
    if changed:
        receipt["nested_metadata_projection"] = {
            "source_revision": BEFORE,
            "scope": "Public derivatives of redacted metadata, including nested canonical digests",
            "measurements_changed": False,
            "private_artifact_hashes_replaced": False,
            "iterations": iteration + 1,
            "files": {n: {"previous_public_sha256": sha(base[n].encode()),
                          "projected_public_sha256": sha(text.encode())} for n, text in changed.items()},
            "canonical_projection": {key: {"public_digest": value, "object_location": locations[key]}
                                     for key, value in mapping.items()},
        }
        for name, text in changed.items():
            (ROOT / name).write_text(text)
        receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    path = ROOT / "README.md"
    text = path.read_text()
    if "corrected click/cart challenger is **rejected**" not in text.lower():
        anchor = "## Review by interest"
        if text.count(anchor) != 1:
            raise ValueError("README insertion point changed")
        text = text.replace(anchor, "The earlier corrected click/cart challenger is **rejected** after replay against the actual deployed comparator. That validation-integrity case remains available alongside the later orders study. [Inspect the correction](research/frontier/11_validation_integrity_reconciliation.md).\n\n" + anchor)
        path.write_text(text)
    print(json.dumps({"projected_files": len(changed), "iterations": iteration + 1,
                      "measurement_changes": 0, "model_changes": 0}))


if __name__ == "__main__":
    main()
