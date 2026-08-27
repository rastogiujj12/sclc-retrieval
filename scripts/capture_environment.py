#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from sclc.config import load_config

PACKAGES = [
    "numpy",
    "pandas",
    "pydantic",
    "torch",
    "transformers",
    "tokenizers",
    "datasets",
    "scipy",
    "typer",
    "rich",
    "tqdm",
]


def run(command: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            command,
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def git_snapshot(root: Path) -> dict[str, Any]:
    return {
        "commit": run(["git", "rev-parse", "HEAD"], root),
        "status_porcelain": run(["git", "status", "--porcelain"], root),
    }


def frozen_inputs(root: Path) -> dict[str, Any]:
    paths = {
        "sample_manifest": Path("data/subsets/selected_documents.csv"),
        "query_types": Path("data/retrieval_units/query_types.csv"),
        "challenge_review": Path("data/subsets_cross_section_challenge/review_decisions.csv"),
        "challenge_documents": Path("data/subsets_cross_section_challenge/selected_documents.csv"),
    }
    return {
        name: {"path": str(path), "sha256": file_sha256(root / path)}
        for name, path in paths.items()
    }


def torch_snapshot() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {"available": False}
    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            gpus.append(torch.cuda.get_device_name(index))
    return {
        "available": True,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "gpus": gpus,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reproducibility/environment.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config if not args.config.is_absolute() else args.config)
    payload = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "torch": torch_snapshot(),
        "repository": git_snapshot(root),
        "frozen_inputs": frozen_inputs(root),
        "dataset": {
            "repo_id": config.dataset.repo_id,
            "subset": config.dataset.subset,
            "splits": config.dataset.splits,
            "parquet_files": config.dataset.parquet_files,
        },
        "models": {
            "granite": config.models.granite.model_dump(),
            "jina": config.models.jina.model_dump(),
        },
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(output)


if __name__ == "__main__":
    main()
