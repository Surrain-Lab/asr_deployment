"""Configuration loading. Single source of truth for every path the system uses."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("ASRBENCH_CONFIG", ROOT / "config.yaml"))


class Config:
    def __init__(self, data: dict):
        self._d = data
        p = data["paths"]
        self.pipeline3 = Path(p["pipeline3"])
        self.vtc1_model = Path(p["vtc1_model"])
        self.vtc2_model = Path(p["vtc2_model"])
        self.runs_root = Path(p["runs_root"])
        self.conda = Path(p["conda"])

        self.env_main = data["envs"]["main"]
        self.env_vtc1 = data["envs"]["vtc1"]

        self.browse_roots = [Path(x) for x in data["browse_roots"]]

        s = data["scheduler"]
        self.gpus = list(s["gpus"])
        self.max_cpu_jobs = int(s["max_cpu_jobs"])
        self.poll_seconds = int(s.get("poll_seconds", 3))

        self.host = data["web"]["host"]
        self.port = int(data["web"]["port"])

        self.db_path = ROOT / "jobs.db"
        self.modules_dir = ROOT / "modules"

    # --- derived paths into the wrapped Pipeline 3.0 tree ---
    @property
    def human_dir(self) -> Path:
        return self.pipeline3 / "Human_Evaluation" / "Human_Evaluation2.0"

    @property
    def vtc1_dir(self) -> Path:
        return self.pipeline3 / "VTC1_Pipeline"

    @property
    def vtc2_dir(self) -> Path:
        return self.pipeline3 / "VTC2_Pipeline"

    @property
    def whisperx_dir(self) -> Path:
        return self.pipeline3 / "Only_Whisperx"

    def is_browsable(self, path: Path) -> bool:
        """True only if path sits under one of the configured browse roots.

        Resolves symlinks first so a link cannot be used to escape the roots.
        """
        try:
            rp = path.resolve()
        except OSError:
            return False
        for root in self.browse_roots:
            try:
                rp.relative_to(root.resolve())
                return True
            except ValueError:
                continue
        return False


def load() -> Config:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"No configuration at {CONFIG_PATH}.\n"
            f"Copy the example and edit it for this machine:\n"
            f"    cp config.example.yaml config.yaml\n"
            f"    python -m asrbench.doctor --suggest"
        )
    with open(CONFIG_PATH) as f:
        data = yaml.safe_load(f)
    missing = [k for k in ("paths", "envs", "browse_roots", "scheduler", "web")
               if k not in data]
    if missing:
        raise ValueError(f"{CONFIG_PATH} is missing section(s): {', '.join(missing)}")
    return Config(data)


cfg = load()
