import json
from pathlib import Path


def test_python_version_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / ".python-version").read_text(encoding="utf-8").strip() == "3.13.15"


def test_gpu_contract_is_cohesive() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (root / "config/runtime/gpu.json").read_text(encoding="utf-8")
    )

    assert contract["python_series"] == "3.13"
    assert contract["torch"] == "2.13.0"
    assert contract["torchrec"] == "1.8.0"
    assert contract["fbgemm_gpu"] == "1.8.0"
    assert contract["cuda"] == "12.6"
