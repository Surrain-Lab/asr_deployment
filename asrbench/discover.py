"""Work out what is actually inside a directory the user picked.

Two tree shapes occur in this project, and telling them apart matters:

  A. input corpus    <root>/<recording>/<clip>.wav          clips are FILES
  B. pipeline output <root>/<recording>/<clip>/<clip>_<SPEAKER>_clean.txt
                                                            clips are FOLDERS

They are genuinely ambiguous by depth alone - both put files three levels down
once a wrapper folder is involved - so shape B is identified by its filenames:
in an output tree every file is the clip name plus a suffix drawn from a small
known vocabulary (clean, tagged, KCHI, FEM, ...), whereas in an input tree the
remainder is a clip index.

Getting this right is not cosmetic. Pipeline 3.0 assumes recordings are
immediate subdirectories of whatever it is handed, so a wrong level yields empty
or mis-nested output with no error. An annotation root holding several wrapper
folders - a corpus delivered in batches, say - is the common trap: selecting the
parent silently merges collections that are kept apart on disk, so Corpus.mixed
reports it rather than guessing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

AUDIO_EXTS = {".wav"}
TEXT_EXTS = {".txt"}

# Suffixes the pipeline appends to a clip name inside an output clip folder.
CLIP_FILE_TOKENS = {
    "clean", "tagged", "langdetect", "with", "SPEECH", "OVERLAP",
    "KCHI", "OCH", "CHI", "MAL", "FEM", "ADULT", "CHILDREN", "COMBINED",
}

SHAPE_INPUT = "input"    # recording/clip.ext
SHAPE_OUTPUT = "output"  # recording/clip/clip_SPEAKER_kind.txt


@dataclass
class Recording:
    name: str
    path: Path
    clips: list[str] = field(default_factory=list)
    n_files: int = 0


@dataclass
class Corpus:
    root: Path
    effective_root: Path
    recordings: list[Recording]
    shape: str = SHAPE_INPUT
    extra_levels: int = 0
    flat: bool = False
    groups: list[str] = field(default_factory=list)   # distinct parents of recordings
    n_files: int = 0

    @property
    def n_recordings(self) -> int:
        return len(self.recordings)

    @property
    def n_clips(self) -> int:
        return sum(len(r.clips) for r in self.recordings)

    @property
    def ok(self) -> bool:
        return self.n_clips > 0

    @property
    def mixed(self) -> bool:
        """Recordings drawn from more than one parent folder."""
        return len(self.groups) > 1

    def note(self) -> str:
        if not self.ok:
            return "No matching files found under this directory."
        bits = [f"{self.n_recordings} recording(s), {self.n_clips} clip(s)"]
        if self.shape == SHAPE_OUTPUT:
            bits.append(f"{self.n_files} file(s) - pipeline output layout")
        if self.flat:
            bits.append("clips sit directly here, treated as one recording")
        elif self.extra_levels:
            bits.append(f"recordings sit {self.extra_levels} level(s) below the folder you picked")
        return "; ".join(bits)

    def as_dict(self) -> dict:
        return {
            "root": str(self.root),
            "effective_root": str(self.effective_root),
            "shape": self.shape,
            "n_recordings": self.n_recordings,
            "n_clips": self.n_clips,
            "n_files": self.n_files,
            "extra_levels": self.extra_levels,
            "flat": self.flat,
            "mixed": self.mixed,
            "groups": self.groups,
            "ok": self.ok,
            "note": self.note(),
            "recordings": [
                {"name": r.name, "n_clips": len(r.clips),
                 "rel": str(r.path.relative_to(self.root)) if _under(r.path, self.root) else str(r.path),
                 "clips": r.clips[:4]}
                for r in self.recordings[:20]
            ],
        }


def _under(p: Path, root: Path) -> bool:
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False


def _common_ancestor(paths: list[Path]) -> Path:
    parts = [p.parts for p in paths]
    common = []
    for chunk in zip(*parts):
        if len(set(chunk)) == 1:
            common.append(chunk[0])
        else:
            break
    return Path(*common) if common else paths[0]


def _is_clip_folder(dirname: str, filenames: list[str]) -> bool:
    """True when files here are '<dirname>_<known suffix>' - an output clip folder."""
    if not filenames:
        return False
    hits = 0
    for fn in filenames:
        stem = Path(fn).stem
        if not stem.startswith(dirname):
            continue
        rest = stem[len(dirname):].strip("_")
        if not rest or CLIP_FILE_TOKENS & set(rest.split("_")):
            hits += 1
    return hits >= max(2, len(filenames) / 2)


def scan(root: str | Path, exts: set[str] = AUDIO_EXTS, limit: int = 400000) -> Corpus:
    root = Path(root)
    if not root.is_dir():
        return Corpus(root, root, [])

    by_parent: dict[Path, list[str]] = {}
    total = 0
    # followlinks=True is required: recording folders are routinely symlinks
    # here (the projects live on an NFS share and are linked into $HOME), and
    # without it a corpus assembled from links scans as empty. The visited set
    # keeps a symlink loop from spinning forever.
    visited: set[tuple[int, int]] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        try:
            st = os.stat(dirpath)
            key = (st.st_dev, st.st_ino)
        except OSError:
            continue
        if key in visited:
            dirnames[:] = []
            continue
        visited.add(key)

        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        keep = [f for f in filenames if Path(f).suffix.lower() in exts]
        if keep:
            by_parent.setdefault(Path(dirpath), []).extend(keep)
            total += len(keep)
        if total >= limit:
            break

    if not by_parent:
        return Corpus(root, root, [])

    leaves = sorted(by_parent)

    # Clips directly in the chosen folder: a single implicit recording.
    if leaves == [root]:
        return Corpus(root, root,
                      [Recording(root.name, root, sorted(by_parent[root]),
                                 len(by_parent[root]))],
                      SHAPE_INPUT, 0, True, [str(root.parent)], total)

    clip_like = sum(1 for d in leaves if _is_clip_folder(d.name, by_parent[d]))
    shape = SHAPE_OUTPUT if clip_like >= max(1, len(leaves) / 2) else SHAPE_INPUT

    if shape == SHAPE_OUTPUT:
        # Leaf folders are clips; their parents are the recordings.
        recs: dict[Path, Recording] = {}
        for leaf in leaves:
            parent = leaf.parent
            r = recs.setdefault(parent, Recording(parent.name, parent))
            r.clips.append(leaf.name)
            r.n_files += len(by_parent[leaf])
        recordings = [recs[k] for k in sorted(recs)]
    else:
        recordings = [Recording(d.name, d, sorted(by_parent[d]), len(by_parent[d]))
                      for d in leaves]

    rec_paths = [r.path for r in recordings]
    groups = sorted({str(p.parent) for p in rec_paths})
    effective = _common_ancestor(rec_paths) if len(rec_paths) > 1 else rec_paths[0].parent

    try:
        extra = len(effective.relative_to(root).parts)
    except ValueError:
        extra = 0

    return Corpus(root, effective, recordings, shape, extra, False, groups, total)


def scan_audio(root) -> Corpus:
    return scan(root, AUDIO_EXTS)


def scan_text(root) -> Corpus:
    return scan(root, TEXT_EXTS)
