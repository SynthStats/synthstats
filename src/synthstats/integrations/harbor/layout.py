"""Resolve Harbor's local trial artifact layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit


@dataclass(frozen=True)
class HarborTrialPaths:
    root: Path

    @classmethod
    def from_uri(cls, uri: str | Path) -> HarborTrialPaths:
        if isinstance(uri, Path):
            return cls(uri)
        parsed = urlsplit(str(uri))
        if parsed.scheme not in {"", "file"}:
            raise ValueError(f"Unsupported Harbor trial URI scheme: {parsed.scheme}")
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError("Harbor file URI must refer to the local filesystem")
        return cls(Path(unquote(parsed.path)) if parsed.scheme else Path(uri))

    @property
    def trajectory(self) -> Path:
        return self.root / "agent" / "trajectory.json"

    @property
    def verifier(self) -> Path:
        return self.root / "verifier"


def reward_directory(trajectory: Path) -> Path:
    """Resolve verifier siblings for Harbor trials and adjacent standalone records."""
    if trajectory.parent.name == "agent":
        return HarborTrialPaths(trajectory.parent.parent).verifier
    return trajectory.parent
