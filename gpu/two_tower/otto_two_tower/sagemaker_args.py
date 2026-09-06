"""Model SageMaker's JSON decoding before serializing scalar worker arguments.

The training toolkit reads each value with json.loads before building argv.
For example, the API string "true" becomes the CLI string "True". CI compares
this dependency-free preflight adapter with the pinned toolkit's real loader,
framework-parameter split, and argument serializer.
"""

from __future__ import annotations

import json
from collections.abc import Mapping


def worker_arguments(parameters: Mapping[str, str], *, program: str) -> list[str]:
    decoded: dict[str, object] = {}
    for name, value in parameters.items():
        if not name or not isinstance(value, str):
            raise ValueError("managed hyperparameters must be named string values")
        try:
            decoded[name] = json.loads(value)
        except (ValueError, TypeError):
            decoded[name] = value
    if decoded.get("sagemaker_program") != program:
        raise ValueError(f"managed entrypoint must be {program}")
    arguments: list[str] = []
    for name, decoded_value in sorted(decoded.items()):
        if name.startswith("sagemaker_"):
            continue
        if decoded_value is not None and not isinstance(decoded_value, (str, bool, int, float)):
            raise ValueError(f"managed worker requires a scalar hyperparameter: {name}")
        option = ("--" if len(name) > 1 else "-") + name
        arguments.extend((option, "" if decoded_value is None else str(decoded_value)))
    return arguments
