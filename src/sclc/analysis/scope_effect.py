from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from sclc.config import AppConfig
from sclc.data.retrieval_unit_io import read_top_level_sections
from sclc.options import EmbeddingModel, RetrievalCondition
from sclc.paths import analysis_dir, evaluation_dir, retrieval_unit_dir


def _fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _evaluation_dir(
    config: AppConfig,
    condition: RetrievalCondition,
    model: EmbeddingModel,
    *,
    chunk_size: int,
) -> Path:
    return evaluation_dir(config, chunk_size) / condition.value / model.value


def _load_evaluation(
    config: AppConfig,
    condition: RetrievalCondition,
    model: EmbeddingModel,
    *,
    chunk_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = _evaluation_dir(config, condition, model, chunk_size=chunk_size)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} does not exist. Evaluate {condition.value}/{model.value} first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics_path = directory / manifest["files"]["query_metrics"]
    frame = pd.read_csv(
        metrics_path,
        dtype={"query_id": "string", "document_id": "string"},
    )
    return frame, manifest


def _section_features(config: AppConfig, *, chunk_size: int) -> pd.DataFrame:
    path = retrieval_unit_dir(config, chunk_size) / "top_level_sections.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run `sclc chunk` first.")
    rows = [
        {
            "document_id": section.document_id,
            "section_characters": section.span.end - section.span.start,
        }
        for section in read_top_level_sections(path)
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No top-level sections are available for scope-effect analysis")
    return (
        frame.groupby("document_id", as_index=False)
        .agg(
            top_level_section_count=("section_characters", "count"),
            mean_section_characters=("section_characters", "mean"),
            maximum_section_characters=("section_characters", "max"),
            total_section_characters=("section_characters", "sum"),
        )
    )


def _document_features(config: AppConfig, *, chunk_size: int) -> pd.DataFrame:
    path = config.paths.subset_dir / "selected_documents.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run `sclc sample` first.")
    frame = pd.read_csv(path, dtype={"document_id": "string"})
    required = {
        "document_id",
        "analysis_set",
        "granite_tokens",
        "jina_tokens",
        "character_count",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame.merge(
        _section_features(config, chunk_size=chunk_size),
        on="document_id",
        how="left",
        validate="one_to_one",
    )


def _fit_univariate(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return {
            "document_count": float(len(x)),
            "pearson_r": float("nan"),
            "slope": float("nan"),
            "intercept": float("nan"),
            "r_squared": float("nan"),
        }
    correlation = float(np.corrcoef(x, y)[0, 1])
    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = intercept + slope * x
    total = float(np.square(y - y.mean()).sum())
    residual = float(np.square(y - prediction).sum())
    r_squared = 1.0 - residual / total if total > 0 else float("nan")
    return {
        "document_count": float(len(x)),
        "pearson_r": correlation,
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
    }


def _analysis_groups(model: EmbeddingModel) -> list[tuple[str, set[str]]]:
    groups = [("cross_model_core", {"cross_model_core"})]
    if model is EmbeddingModel.GRANITE:
        groups.extend(
            [
                ("granite_extended", {"granite_extended"}),
                ("all_eligible", {"cross_model_core", "granite_extended"}),
            ]
        )
    return groups


def analyse_scope_effect(
    config: AppConfig,
    *,
    chunk_size: int,
    models: Sequence[EmbeddingModel] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Analyse section-constrained minus global retrieval performance.

    The exploratory analysis is document-level: query effects are macro-averaged
    within each paper before their association with document length and structure
    is described. This prevents papers with more QASPER questions from dominating
    the structural analysis.
    """
    selected_models = tuple(
        dict.fromkeys(tuple(EmbeddingModel) if models is None else models)
    )
    if not selected_models:
        raise ValueError("At least one embedding model must be selected")
    unknown_models = set(selected_models).difference(EmbeddingModel)
    if unknown_models:
        raise ValueError(f"Unknown embedding models: {sorted(unknown_models)}")

    output_dir = analysis_dir(config, chunk_size) / "scope_effect"
    if set(selected_models) != set(EmbeddingModel):
        output_dir = output_dir / "__".join(model.value for model in selected_models)
    query_path = output_dir / "scope_effect_queries.csv"
    document_path = output_dir / "scope_effect_documents.csv"
    associations_path = output_dir / "scope_effect_associations.csv"
    manifest_path = output_dir / "manifest.json"

    metric = config.evaluation.primary_metric
    source_manifests: dict[str, str] = {}
    query_frames: list[pd.DataFrame] = []
    for model in selected_models:
        section, section_manifest = _load_evaluation(
            config, RetrievalCondition.SECTION_CONSTRAINED, model, chunk_size=chunk_size
        )
        global_frame, global_manifest = _load_evaluation(
            config, RetrievalCondition.GLOBAL, model, chunk_size=chunk_size
        )
        for name, frame in (("section_constrained", section), ("global", global_frame)):
            if "split" not in frame.columns:
                raise ValueError(f"Evaluation {name}/{model.value} is missing split")
        section = section[
            section["split"] == config.evaluation.confirmatory_split
        ].copy()
        global_frame = global_frame[
            global_frame["split"] == config.evaluation.confirmatory_split
        ].copy()
        if section.empty or global_frame.empty:
            raise RuntimeError(
                f"No {config.evaluation.confirmatory_split} queries for {model.value}"
            )
        for name, frame in (("section_constrained", section), ("global", global_frame)):
            required = {
                "query_id",
                "document_id",
                "analysis_set",
                "query_type",
                "question",
                metric,
            }
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(
                    f"Evaluation {name}/{model.value} is missing columns: {sorted(missing)}"
                )
        source_manifests[f"section_constrained/{model.value}"] = section_manifest[
            "configuration_fingerprint"
        ]
        source_manifests[f"global/{model.value}"] = global_manifest[
            "configuration_fingerprint"
        ]
        merged = section[
            ["query_id", "document_id", "analysis_set", "query_type", "question", metric]
        ].merge(
            global_frame[["query_id", "document_id", metric]],
            on=["query_id", "document_id"],
            how="inner",
            validate="one_to_one",
            suffixes=("_section_constrained", "_global"),
        )
        if len(merged) != len(section) or len(merged) != len(global_frame):
            raise RuntimeError(
                f"Section-constrained and global query sets differ for {model.value}"
            )
        merged["model_key"] = model.value
        merged["metric"] = metric
        merged["section_constrained_score"] = merged[f"{metric}_section_constrained"]
        merged["global_score"] = merged[f"{metric}_global"]
        merged["scope_effect_section_minus_global"] = (
            merged["section_constrained_score"] - merged["global_score"]
        )
        query_frames.append(
            merged[
                [
                    "model_key",
                    "query_id",
                    "document_id",
                    "analysis_set",
                    "query_type",
                    "question",
                    "metric",
                    "section_constrained_score",
                    "global_score",
                    "scope_effect_section_minus_global",
                ]
            ]
        )

    features = _document_features(config, chunk_size=chunk_size)
    query_frame = pd.concat(query_frames, ignore_index=True)
    document_frame = (
        query_frame.groupby(["model_key", "document_id", "analysis_set"], as_index=False)
        .agg(
            query_count=("query_id", "count"),
            mean_section_constrained_score=("section_constrained_score", "mean"),
            mean_global_score=("global_score", "mean"),
            mean_scope_effect_section_minus_global=(
                "scope_effect_section_minus_global",
                "mean",
            ),
        )
        .merge(features, on=["document_id", "analysis_set"], how="left", validate="many_to_one")
    )
    if document_frame[
        ["granite_tokens", "jina_tokens", "top_level_section_count"]
    ].isna().any().any():
        raise RuntimeError("Document features could not be matched to every evaluated paper")

    association_rows: list[dict[str, Any]] = []
    predictors = [
        "character_count",
        "top_level_section_count",
        "mean_section_characters",
        "maximum_section_characters",
    ]
    for model in selected_models:
        model_frame = document_frame[document_frame["model_key"] == model.value]
        token_predictor = "granite_tokens" if model is EmbeddingModel.GRANITE else "jina_tokens"
        for analysis_name, analysis_sets in _analysis_groups(model):
            group = model_frame[model_frame["analysis_set"].isin(analysis_sets)]
            for predictor in [token_predictor, *predictors]:
                result = _fit_univariate(
                    group[predictor].to_numpy(dtype=float),
                    group["mean_scope_effect_section_minus_global"].to_numpy(dtype=float),
                )
                association_rows.append(
                    {
                        "model_key": model.value,
                        "analysis_set": analysis_name,
                        "metric": metric,
        "chunk_size_tokens": chunk_size,
        "confirmatory_split": config.evaluation.confirmatory_split,
                        "predictor": predictor,
                        **result,
                    }
                )
    associations = pd.DataFrame(association_rows)

    configuration = {
        "models": [model.value for model in selected_models],
        "metric": metric,
        "effect": "section_constrained_minus_global",
        "aggregation": "mean_within_document_before_structural_association",
        "source_manifests": source_manifests,
        "feature_documents": features[
            [
                "document_id",
                "analysis_set",
                "granite_tokens",
                "jina_tokens",
                "character_count",
                "top_level_section_count",
                "mean_section_characters",
                "maximum_section_characters",
            ]
        ].to_dict(orient="records"),
    }
    fingerprint = _fingerprint(configuration)
    if manifest_path.exists() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("configuration_fingerprint") != fingerprint:
            raise RuntimeError(
                f"Cached scope-effect analysis at {output_dir} does not match current "
                "inputs. Re-run with --overwrite."
            )
        return existing

    output_dir.mkdir(parents=True, exist_ok=True)
    query_frame.sort_values(["model_key", "analysis_set", "document_id", "query_id"]).to_csv(
        query_path, index=False
    )
    document_frame.sort_values(["model_key", "analysis_set", "document_id"]).to_csv(
        document_path, index=False
    )
    associations.sort_values(["model_key", "analysis_set", "predictor"]).to_csv(
        associations_path, index=False
    )
    manifest = {
        "schema_version": 1,
        "kind": "scope_effect_analysis",
        "chunk_size_tokens": chunk_size,
        "metric": metric,
        "query_count": len(query_frame),
        "document_rows": len(document_frame),
        "configuration": configuration,
        "configuration_fingerprint": fingerprint,
        "files": {
            "query_effects": query_path.name,
            "document_effects": document_path.name,
            "associations": associations_path.name,
        },
        "interpretation_note": (
            "Associations are exploratory descriptive Pearson correlations and "
            "univariate OLS fits; they are not confirmatory causal estimates."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
