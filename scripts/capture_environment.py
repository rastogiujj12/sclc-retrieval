#!/usr/bin/env python3
"""Capture the local software, CUDA, Git, and Hugging Face cache snapshot.

This script is intentionally offline. If the model snapshots already exist in the
configured Hugging Face cache, their cached commit references are recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from sclc.config import load_config

PACKAGE_NAMES = {
    "datasets": "datasets",
    "huggingface_hub": "huggingface-hub",
    "numpy": "numpy",
    "pandas": "pandas",
    "peft": "peft",
    "pyarrow": "pyarrow",
    "pydantic": "pydantic",
    "PyYAML": "PyYAML",
    "rich": "rich",
    "torch": "torch",
    "tqdm": "tqdm",
    "transformers": "transformers",
    "typer": "typer",
}


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_input_snapshot(repository_root: Path) -> dict[str, dict[str, str | None]]:
    relative_paths = {
        "sample_manifest": Path("data/subsets/selected_documents.csv"),
        "query_types": Path("data/retrieval_units/query_types.csv"),
        "query_type_coding_record": Path(
            "data/retrieval_units/query_type_coding_record.csv"
        ),
        "challenge_review": Path(
            "data/subsets_cross_section_challenge/review_decisions.csv"
        ),
        "challenge_documents": Path(
            "data/subsets_cross_section_challenge/selected_documents.csv"
        ),
    }
    return {
        label: {
            "path": str(relative_path),
            "sha256": file_sha256(repository_root / relative_path),
        }
        for label, relative_path in relative_paths.items()
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for label, distribution in PACKAGE_NAMES.items():
        try:
            versions[label] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[label] = None
    return versions


def git_snapshot(repository_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=repository_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None

    commit = run("rev-parse", "HEAD")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "branch": branch,
        "working_tree_clean": None if status is None else not bool(status),
    }


def cached_hf_revision(cache_dir: Path, repo_id: str) -> str | None:
    repo_cache = cache_dir / f"models--{repo_id.replace('/', '--')}"
    main_ref = repo_cache / "refs" / "main"
    if main_ref.exists():
        value = main_ref.read_text(encoding="utf-8").strip()
        return value or None

    snapshots = repo_cache / "snapshots"
    if snapshots.exists():
        candidates = sorted(path.name for path in snapshots.iterdir() if path.is_dir())
        if len(candidates) == 1:
            return candidates[0]
    return None


def cuda_snapshot() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch_available": False}

    snapshot: dict[str, Any] = {
        "torch_available": True,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        ),
        "devices": [],
    }
    if torch.cuda.is_available():
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
        snapshot["devices"] = devices
    return snapshot


def build_snapshot(config_path: Path, repository_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    cache_dir = config.paths.hf_cache_dir

    models: dict[str, Any] = {}
    for name in ("granite", "jina"):
        model = getattr(config.models, name)
        models[name] = {
            "model_id": model.model_id,
            "configured_revision": model.revision,
            "cached_main_revision": cached_hf_revision(cache_dir, model.model_id),
            "adapter_source": model.adapter_source,
            "adapter_revision": model.adapter_revision,
            "passage_adapter": model.passage_adapter,
            "query_adapter": model.query_adapter,
        }

    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": git_snapshot(repository_root),
        "frozen_inputs": frozen_input_snapshot(repository_root),
        "dataset": {
            "repo_id": config.dataset.repo_id,
            "subset": config.dataset.subset,
            "splits": config.dataset.splits,
            "parquet_files": config.dataset.parquet_files,
        },
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "packages": package_versions(),
        "cuda": cuda_snapshot(),
        "configuration": {
            "path": str(config_path),
            "project_seed": config.project.seed,
            "canonical_tokenizer": config.chunking.canonical_tokenizer,
            "canonical_tokenizer_configured_revision": (
                config.chunking.canonical_tokenizer_revision
            ),
            "canonical_tokenizer_cached_main_revision": cached_hf_revision(
                cache_dir, config.chunking.canonical_tokenizer
            ),
            "retrieval_unit_sizes": config.chunking.supported_chunk_sizes,
            "overlap_tokens": config.chunking.overlap_tokens,
            "dense_dtype": config.dense.dtype,
            "dense_output_dtype": config.dense.output_dtype,
            "attention_implementation": config.dense.attn_implementation,
        },
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reproducibility/environment.json"),
    )
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    snapshot = build_snapshot(args.config, repository_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote reproducibility snapshot to {args.output}")


if __name__ == "__main__":
    main()
