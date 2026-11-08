"""Freeze the retraining protocol, and refuse the test split until it is frozen.

The stamp records the SHA-256 of the protocol document and of ``retraining.yaml`` (which holds
every threshold) at the moment the protocol had no open ``[TBD]`` left. The final test stage
checks both hashes again: if either file was edited afterwards, the test split stays closed.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from src.v1_mnist.component.utils.retraining_settings import ExperimentSettings

OPEN_MARKER = "[TBD]"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_stamp(settings: ExperimentSettings) -> Path:
    """Write the stamp. Refuses while a ``[TBD]`` remains, or if the stamp exists already."""
    if settings.stamp_path.exists():
        raise FileExistsError(f"{settings.stamp_path} exists: the protocol is already frozen.")
    open_lines = [
        f"line {i}: {line.strip()[:80]}"
        for i, line in enumerate(settings.protocol_path.read_text(encoding="utf-8").splitlines(), 1)
        if OPEN_MARKER in line
    ]
    if open_lines:
        raise ValueError("The protocol still has open items:\n" + "\n".join(open_lines))
    if settings.macro_f1_guard is None:
        raise ValueError("evaluation.macro_f1_guard in retraining.yaml is still empty.")
    settings.stamp_path.write_text(
        "status: protocol complete; written before the test split was used for Phase 4\n"
        f"stamped_utc: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        f"protocol: {settings.protocol_path.name}\n"
        f"protocol_sha256: {_sha256(settings.protocol_path)}\n"
        f"config: {settings.config_path.name}\n"
        f"config_sha256: {_sha256(settings.config_path)}\n"
        f"macro_f1_guard: {settings.macro_f1_guard}\n",
        encoding="utf-8",
    )
    return settings.stamp_path


def verify_stamp(settings: ExperimentSettings) -> None:
    """Stop unless the stamp exists and both files still match it."""
    if not settings.stamp_path.is_file():
        raise PermissionError(
            "The protocol is not frozen (no stamp), so the test split stays closed. Fill every "
            f"{OPEN_MARKER}, then run the stamp step."
        )
    stamp = dict(
        line.split(": ", 1) for line in settings.stamp_path.read_text(encoding="utf-8").splitlines() if ": " in line
    )
    for label, path, key in (
        ("protocol", settings.protocol_path, "protocol_sha256"),
        ("config", settings.config_path, "config_sha256"),
    ):
        if _sha256(path) != stamp.get(key):
            raise PermissionError(
                f"The {label} file changed after it was frozen ({path.name}). The test split stays "
                "closed; record the change as an amendment and re-freeze under a new revision."
            )
