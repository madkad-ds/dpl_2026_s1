"""
Project configuration: one TOML file at the project root, read once.

The project root is the directory holding `config.toml`; it is located by walking up
from this file, so scripts and notebooks resolve the same paths whether they are run
from the root or from inside a scripts folder. Data and output paths are relative to
that root.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


CONFIG_FILENAME = "config.toml"


def project_root() -> Path:
    """The directory containing config.toml, found by walking up from this file."""
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / CONFIG_FILENAME).exists():
            return candidate
    raise FileNotFoundError(f"{CONFIG_FILENAME} not found above {here}")


@dataclass(frozen=True)
class Settings:
    """Typed view of config.toml plus resolved paths."""

    raw: dict
    root: Path

    # Sections returned as plain dicts so a caller can pass them straight into a
    # sidecar; typed accessors below cover the values scripts read most.
    def section(self, name: str) -> dict:
        return dict(self.raw[name])

    def seed(self, analysis: str) -> int:
        """Fixed seed for one analysis; falls back to the global seed."""
        seeds = self.raw["seeds"]
        return int(seeds.get(analysis, seeds["global"]))

    @property
    def earlier_season(self) -> int:
        return int(self.raw["seasons"]["earlier"])

    @property
    def later_season(self) -> int:
        return int(self.raw["seasons"]["later"])

    def phase(self, name: str) -> tuple[int, int]:
        lo, hi = self.raw["phases"][name]
        return int(lo), int(hi)

    @property
    def n_resamples(self) -> int:
        return int(self.raw["bootstrap"]["n_resamples"])

    @property
    def interval_level(self) -> float:
        return float(self.raw["bootstrap"]["interval_level"])

    @property
    def equivalence_level(self) -> float:
        return float(self.raw["bootstrap"]["equivalence_level"])

    @property
    def data_dir(self) -> Path:
        return self.root / self.raw["paths"]["data_dir"]

    @property
    def outputs_dir(self) -> Path:
        return self.root / self.raw["paths"]["outputs_dir"]

    def data_path(self, key: str) -> Path:
        """Path of a dataset named in [paths], e.g. 'cleaned_dataset'."""
        return self.data_dir / self.raw["paths"][key]

    def output_dir(self, analysis: str) -> Path:
        """Output folder for one analysis, created on first use."""
        folder = self.outputs_dir / analysis
        folder.mkdir(parents=True, exist_ok=True)
        return folder


def load_settings() -> Settings:
    root = project_root()
    with open(root / CONFIG_FILENAME, "rb") as handle:
        raw = tomllib.load(handle)
    return Settings(raw=raw, root=root)
