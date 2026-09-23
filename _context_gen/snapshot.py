# _context_gen/snapshot.py
"""
Snapshot identity.

Two distinct fields, two distinct jobs (implementation contract §15,
architecture contract §11):

  REPOSITORY IDENTITY  commit + dirty flag — coarse, human-facing, answers
                        "roughly what state was this generated from."

  ARTIFACT CONTENT HASH  a hash over the exact (path, content) pairs one
                          specific artifact contains — precise, machine-
                          facing, answers "did THIS artifact's actual
                          content change," independent of unrelated dirty
                          edits elsewhere in the working tree.

No timestamp is recorded anywhere in generated output: a timestamp would
change on every run even with zero content difference, which is exactly
the unnecessary-churn failure mode the determinism requirement (§30)
forbids.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True, kw_only=True)
class RepositoryIdentity:
    commit: str | None  # short hash, or None if not a git repository
    dirty: bool | None  # None if not a git repository


def repository_identity(repo_root: Path) -> RepositoryIdentity:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout
        return RepositoryIdentity(commit=commit, dirty=bool(status.strip()))
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return RepositoryIdentity(commit=None, dirty=None)


def artifact_content_hash(files: list[tuple[str, str]]) -> str:
    """
    Deterministic SHA-256 over sorted (repo_relative_path, content) pairs.
    Two runs against an unchanged file set produce the same hash regardless
    of unrelated dirty edits elsewhere in the tree; a changed file set
    (even one file, even one byte) changes the hash.
    """
    h = hashlib.sha256()
    for path, content in sorted(files, key=lambda pc: pc[0]):
        h.update(path.encode("utf-8"))
        h.update(b"\0")
        h.update(content.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]
