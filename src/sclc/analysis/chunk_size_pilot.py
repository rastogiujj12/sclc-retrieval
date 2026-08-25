from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import pandas as pd

from sclc.config import AppConfig
from sclc.evaluation.statistics import paired_document_bootstrap
from sclc.options import EmbeddingModel, RetrievalCondition
from sclc.paths import evaluation_dir


def _fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _metrics_path(
    config: AppConfig,
    *,
    chunk_size: int,
    condition: RetrievalCondition,
    model: EmbeddingModel | None,
) -> tuple[Path, dict[str, Any]]:
    base = evaluation_dir(config, chunk_size)
    directory = (
        base / condition.value
        if model is None
        else base / condition.value / model.value
    )
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        label = condition.value if model is None else f"{condition.value}/{model.value}"
        raise FileNotFoundError(
            f"Missing {chunk_size}-token evaluation for {label}: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return directory / manifest["files"]["query_metrics"], manifest


def _load_validation_metrics(
    config: AppConfig,
    *,
    chunk_size: int,
    condition: RetrievalCondition,
    model: EmbeddingModel | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path, manifest = _metrics_path(
        config,
        chunk_size=chunk_size,
        condition=condition,
        model=model,
    )
    frame = pd.read_csv(
        path,
        dtype={"query_id": "string", "document_id": "string"},
    )
    if "split" not in frame.columns:
        raise ValueError(f"{path} has no split column")
    validation = frame[frame["split"] == config.pilot.selection_split].copy()
    if validation.empty:
        raise RuntimeError(
            f"{path} contains no {config.pilot.selection_split!r} queries"
        )
    return validation, manifest


def _select_size(summary: pd.DataFrame, config: AppConfig) -> tuple[int, dict[str, Any]]:
    fixed = summary[summary["method"] == "fixed_dense_granite"].copy()
    if set(fixed["chunk_size_tokens"].astype(int)) != set(config.pilot.chunk_sizes):
        raise RuntimeError("Fixed-dense pilot summary is missing one or more chunk sizes")

    margin = config.pilot.practical_equivalence_margin
    primary = config.pilot.primary_metric
    secondary = config.pilot.secondary_metric
    best_primary = float(fixed[primary].max())
    primary_eligible = fixed[fixed[primary] >= best_primary - margin].copy()
    best_secondary = float(primary_eligible[secondary].max())
    secondary_eligible = primary_eligible[
        primary_eligible[secondary] >= best_secondary - margin
    ].copy()
    selected = int(secondary_eligible["chunk_size_tokens"].max())
    details = {
        "best_primary_mean": best_primary,
        "primary_eligible_chunk_sizes": sorted(
            primary_eligible["chunk_size_tokens"].astype(int).tolist()
        ),
        "best_secondary_mean_among_primary_eligible": best_secondary,
        "secondary_eligible_chunk_sizes": sorted(
            secondary_eligible["chunk_size_tokens"].astype(int).tolist()
        ),
        "efficiency_tiebreak": "largest_chunk_size",
    }
    return selected, details


def analyse_chunk_size_pilot(
    config: AppConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir = config.paths.analysis_dir / "chunk_size_pilot"
    summary_path = output_dir / "validation_summary.csv"
    bootstrap_path = output_dir / "pairwise_bootstrap.csv"
    selection_path = output_dir / "selection.json"
    manifest_path = output_dir / "manifest.json"

    method_specs = (
        ("bm25", RetrievalCondition.BM25, None),
        (
            "fixed_dense_granite",
            RetrievalCondition.FIXED_DENSE,
            EmbeddingModel.GRANITE,
        ),
    )
    metrics = [
        config.pilot.primary_metric,
        config.pilot.secondary_metric,
        *config.pilot.robustness_metrics,
    ]
    metrics = list(dict.fromkeys(metrics))

    loaded: dict[tuple[str, int], pd.DataFrame] = {}
    source_fingerprints: dict[str, str] = {}
    expected_query_ids: set[str] | None = None
    for method_name, condition, model in method_specs:
        for chunk_size in config.pilot.chunk_sizes:
            frame, manifest = _load_validation_metrics(
                config,
                chunk_size=chunk_size,
                condition=condition,
                model=model,
            )
            missing = set(metrics).difference(frame.columns)
            if missing:
                raise ValueError(
                    f"Evaluation for {method_name}/{chunk_size} is missing metrics: "
                    f"{sorted(missing)}"
                )
            query_ids = set(frame["query_id"].astype(str))
            if expected_query_ids is None:
                expected_query_ids = query_ids
            elif query_ids != expected_query_ids:
                raise RuntimeError(
                    "Chunk-size pilot inputs do not contain identical validation queries"
                )
            loaded[(method_name, chunk_size)] = frame
            source_fingerprints[f"{method_name}/chunk_{chunk_size}"] = manifest[
                "configuration_fingerprint"
            ]

    configuration = {
        "chunk_sizes": config.pilot.chunk_sizes,
        "selection_split": config.pilot.selection_split,
        "selection_method": "fixed_dense_granite",
        "primary_metric": config.pilot.primary_metric,
        "secondary_metric": config.pilot.secondary_metric,
        "robustness_metrics": config.pilot.robustness_metrics,
        "practical_equivalence_margin": config.pilot.practical_equivalence_margin,
        "bootstrap_iterations": config.evaluation.bootstrap_iterations,
        "confidence_level": config.evaluation.confidence_level,
        "seed": config.project.seed,
        "source_fingerprints": source_fingerprints,
    }
    fingerprint = _fingerprint(configuration)
    if manifest_path.exists() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("configuration_fingerprint") != fingerprint:
            raise RuntimeError(
                "Cached chunk-size pilot does not match current inputs. "
                "Re-run with --overwrite."
            )
        return existing

    summary_rows: list[dict[str, Any]] = []
    for method_name, _, _ in method_specs:
        for chunk_size in config.pilot.chunk_sizes:
            frame = loaded[(method_name, chunk_size)]
            row: dict[str, Any] = {
                "method": method_name,
                "chunk_size_tokens": chunk_size,
                "split": config.pilot.selection_split,
                "query_count": len(frame),
                "document_count": int(frame["document_id"].nunique()),
                "mean_candidate_count": float(frame["candidate_count"].mean()),
                "proportion_saturated_at_5": float(
                    frame["cutoff_saturated_at_5"].mean()
                ),
                "proportion_saturated_at_10": float(
                    frame["cutoff_saturated_at_10"].mean()
                ),
            }
            for metric in metrics:
                row[metric] = float(frame[metric].mean())
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["method", "chunk_size_tokens"]
    )

    bootstrap_rows: list[dict[str, Any]] = []
    seed_offset = 0
    for method_name, _, _ in method_specs:
        for first_size, second_size in itertools.combinations(
            config.pilot.chunk_sizes, 2
        ):
            first = loaded[(method_name, first_size)]
            second = loaded[(method_name, second_size)]
            metadata = ["query_id", "document_id"]
            merged = first[metadata + metrics].merge(
                second[metadata + metrics],
                on=metadata,
                suffixes=("__first", "__second"),
                validate="one_to_one",
            )
            if len(merged) != len(first) or len(merged) != len(second):
                raise RuntimeError("Pilot query sets differ between chunk sizes")
            for metric in metrics:
                seed_offset += 1
                observed, lower, upper, p_value, _ = paired_document_bootstrap(
                    merged,
                    first_column=f"{metric}__first",
                    second_column=f"{metric}__second",
                    iterations=config.evaluation.bootstrap_iterations,
                    confidence_level=config.evaluation.confidence_level,
                    seed=config.project.seed + seed_offset,
                )
                bootstrap_rows.append(
                    {
                        "method": method_name,
                        "split": config.pilot.selection_split,
                        "first_chunk_size": first_size,
                        "second_chunk_size": second_size,
                        "metric": metric,
                        "query_count": len(merged),
                        "document_count": int(merged["document_id"].nunique()),
                        "mean_difference_second_minus_first": observed,
                        "confidence_lower": lower,
                        "confidence_upper": upper,
                        "bootstrap_p_value": p_value,
                    }
                )
    bootstrap = pd.DataFrame(bootstrap_rows)
    selected, selection_details = _select_size(summary, config)

    selection = {
        "selected_chunk_size_tokens": selected,
        "selection_split": config.pilot.selection_split,
        "test_split_used_for_selection": False,
        "selection_method": "fixed_dense_granite",
        "primary_metric": config.pilot.primary_metric,
        "secondary_metric": config.pilot.secondary_metric,
        "practical_equivalence_margin": config.pilot.practical_equivalence_margin,
        "rule": [
            "Retain sizes within the practical-equivalence margin of the best primary mean.",
            "Among those, retain sizes within the margin of the best secondary mean.",
            "Choose the largest remaining chunk size as the efficiency tie-breaker.",
        ],
        "details": selection_details,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "kind": "chunk_size_pilot",
        "selected_chunk_size_tokens": selected,
        "validation_query_count": len(expected_query_ids or set()),
        "configuration": configuration,
        "configuration_fingerprint": fingerprint,
        "files": {
            "validation_summary": summary_path.name,
            "pairwise_bootstrap": bootstrap_path.name,
            "selection": selection_path.name,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


__all__ = ["analyse_chunk_size_pilot"]
