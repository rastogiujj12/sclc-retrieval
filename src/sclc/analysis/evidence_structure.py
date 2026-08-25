from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sclc.config import AppConfig
from sclc.data.io import read_documents_jsonl
from sclc.data.query_types import load_query_types
from sclc.data.retrieval_unit_io import read_prepared_queries, read_top_level_sections
from sclc.data.schema import PreparedQueryRecord
from sclc.evaluation.statistics import holm_adjust, paired_document_bootstrap
from sclc.options import EmbeddingModel, RetrievalCondition
from sclc.paths import analysis_dir, evaluation_dir, retrieval_unit_dir


SINGLE_PARAGRAPH = "single_paragraph"
MULTI_PARAGRAPH_SAME_SECTION = "multi_paragraph_same_section"
MULTI_PARAGRAPH_CROSS_SECTION = "multi_paragraph_cross_section"

EVIDENCE_STRUCTURES: tuple[str, ...] = (
    SINGLE_PARAGRAPH,
    MULTI_PARAGRAPH_SAME_SECTION,
    MULTI_PARAGRAPH_CROSS_SECTION,
)

_STRUCTURE_ORDER = {label: index for index, label in enumerate(EVIDENCE_STRUCTURES)}

# These comparisons preserve identical section-bounded target spans. They isolate
# the effect of increasing contextual encoding scope from chunk -> section -> paper.
SCOPE_COMPARISONS: tuple[
    tuple[RetrievalCondition, RetrievalCondition], ...
] = (
    (
        RetrievalCondition.SECTION_ISOLATED,
        RetrievalCondition.SECTION_CONSTRAINED,
    ),
    (RetrievalCondition.SECTION_CONSTRAINED, RetrievalCondition.GLOBAL),
    (RetrievalCondition.SECTION_ISOLATED, RetrievalCondition.GLOBAL),
)

SUMMARY_CONDITIONS: tuple[RetrievalCondition, ...] = (
    RetrievalCondition.BM25,
    RetrievalCondition.FIXED_DENSE,
    RetrievalCondition.SECTION_ISOLATED,
    RetrievalCondition.SECTION_CONSTRAINED,
    RetrievalCondition.GLOBAL,
)


def _fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _structure_for_set(
    paragraph_ids: Sequence[str],
    paragraph_to_parent_section: Mapping[str, str],
) -> tuple[str, int, int]:
    unique_ids = sorted(set(paragraph_ids))
    if not unique_ids:
        raise ValueError("An acceptable evidence set cannot be empty")

    missing = [
        paragraph_id
        for paragraph_id in unique_ids
        if paragraph_id not in paragraph_to_parent_section
    ]
    if missing:
        raise ValueError(
            "Evidence paragraphs could not be mapped to top-level sections: "
            f"{missing[:10]}"
        )

    parent_sections = {
        paragraph_to_parent_section[paragraph_id] for paragraph_id in unique_ids
    }
    paragraph_count = len(unique_ids)
    section_count = len(parent_sections)

    if paragraph_count == 1:
        structure = SINGLE_PARAGRAPH
    elif section_count == 1:
        structure = MULTI_PARAGRAPH_SAME_SECTION
    else:
        structure = MULTI_PARAGRAPH_CROSS_SECTION
    return structure, paragraph_count, section_count


def classify_evidence_structure(
    query: PreparedQueryRecord,
    paragraph_to_parent_section: Mapping[str, str],
) -> dict[str, Any]:
    """Classify a query by the least-distributed minimal acceptable support set.

    QASPER may provide multiple alternative acceptable evidence sets. Complete-
    support evaluation succeeds when any one set is fully recovered, so the
    primary structure is based on the smallest complete acceptable set. If tied
    minimal sets have different structures, the least-distributed structure is
    selected conservatively and the ambiguity is retained explicitly.
    """

    if not query.evidence_sets:
        raise ValueError(f"Query {query.query_id} has no acceptable evidence sets")

    set_rows: list[dict[str, Any]] = []
    for evidence_set in query.evidence_sets:
        structure, paragraph_count, section_count = _structure_for_set(
            evidence_set.paragraph_ids,
            paragraph_to_parent_section,
        )
        set_rows.append(
            {
                "evidence_set_id": evidence_set.evidence_set_id,
                "paragraph_count": paragraph_count,
                "top_level_section_count": section_count,
                "evidence_structure": structure,
            }
        )

    minimum_paragraph_count = min(row["paragraph_count"] for row in set_rows)
    minimal_rows = [
        row for row in set_rows if row["paragraph_count"] == minimum_paragraph_count
    ]
    minimal_structures = sorted(
        {row["evidence_structure"] for row in minimal_rows},
        key=_STRUCTURE_ORDER.__getitem__,
    )
    primary_structure = minimal_structures[0]
    primary_candidates = [
        row
        for row in minimal_rows
        if row["evidence_structure"] == primary_structure
    ]
    primary_evidence_set_id = sorted(
        str(row["evidence_set_id"]) for row in primary_candidates
    )[0]

    union_structure, union_paragraph_count, union_section_count = _structure_for_set(
        query.evidence_union_paragraph_ids,
        paragraph_to_parent_section,
    )
    all_structures = sorted(
        {row["evidence_structure"] for row in set_rows},
        key=_STRUCTURE_ORDER.__getitem__,
    )

    return {
        "evidence_structure": primary_structure,
        "primary_evidence_set_id": primary_evidence_set_id,
        "minimum_evidence_paragraph_count": minimum_paragraph_count,
        "minimal_evidence_set_count": len(minimal_rows),
        "minimum_top_level_section_count_among_minimal_sets": min(
            row["top_level_section_count"] for row in minimal_rows
        ),
        "maximum_top_level_section_count_among_minimal_sets": max(
            row["top_level_section_count"] for row in minimal_rows
        ),
        "minimal_structure_variants": "|".join(minimal_structures),
        "minimal_structure_ambiguous": len(minimal_structures) > 1,
        "cross_section_required_among_minimal_sets": all(
            row["evidence_structure"] == MULTI_PARAGRAPH_CROSS_SECTION
            for row in minimal_rows
        ),
        "cross_section_required": all(
            row["evidence_structure"] == MULTI_PARAGRAPH_CROSS_SECTION
            for row in set_rows
        ),
        "cross_section_possible": any(
            row["evidence_structure"] == MULTI_PARAGRAPH_CROSS_SECTION
            for row in set_rows
        ),
        "same_section_multi_paragraph_possible": any(
            row["evidence_structure"] == MULTI_PARAGRAPH_SAME_SECTION
            for row in set_rows
        ),
        "single_paragraph_possible": any(
            row["evidence_structure"] == SINGLE_PARAGRAPH for row in set_rows
        ),
        "acceptable_set_structures": "|".join(all_structures),
        "evidence_set_count": len(set_rows),
        "evidence_union_structure": union_structure,
        "evidence_union_paragraph_count": union_paragraph_count,
        "evidence_union_top_level_section_count": union_section_count,
        "evidence_set_details_json": json.dumps(set_rows, sort_keys=True),
    }


def _build_paragraph_parent_map(
    config: AppConfig,
    *,
    chunk_size: int,
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    documents_path = config.paths.processed_dir / "documents.jsonl"
    top_sections_path = retrieval_unit_dir(config, chunk_size) / "top_level_sections.jsonl"
    if not documents_path.exists():
        raise FileNotFoundError(
            f"{documents_path} does not exist. Run `sclc prepare` first."
        )
    if not top_sections_path.exists():
        raise FileNotFoundError(
            f"{top_sections_path} does not exist. "
            f"Run `sclc chunk --chunk-size {chunk_size}` first."
        )

    documents = {
        document.document_id: document
        for document in read_documents_jsonl(documents_path)
    }
    section_to_parent: dict[tuple[str, str], str] = {}
    parent_headings: dict[str, str] = {}
    selected_document_ids: set[str] = set()
    for parent in read_top_level_sections(top_sections_path):
        selected_document_ids.add(parent.document_id)
        parent_headings[parent.parent_section_id] = parent.heading
        for source_section_id in parent.source_section_ids:
            key = (parent.document_id, source_section_id)
            if key in section_to_parent:
                raise RuntimeError(
                    "A source section maps to multiple top-level parents: "
                    f"{parent.document_id}/{source_section_id}"
                )
            section_to_parent[key] = parent.parent_section_id

    missing_documents = selected_document_ids.difference(documents)
    if missing_documents:
        raise RuntimeError(
            "Selected documents are missing from documents.jsonl: "
            f"{sorted(missing_documents)[:10]}"
        )

    paragraph_maps: dict[str, dict[str, str]] = {}
    for document_id in sorted(selected_document_ids):
        document = documents[document_id]
        paragraph_map: dict[str, str] = {}
        for paragraph in document.paragraphs:
            key = (document_id, paragraph.section_id)
            parent_section_id = section_to_parent.get(key)
            if parent_section_id is None:
                raise RuntimeError(
                    "Paragraph section could not be mapped to a top-level section: "
                    f"{document_id}/{paragraph.paragraph_id}/{paragraph.section_id}"
                )
            paragraph_map[paragraph.paragraph_id] = parent_section_id
        paragraph_maps[document_id] = paragraph_map
    return paragraph_maps, parent_headings


def build_evidence_structure_frame(
    config: AppConfig,
    *,
    chunk_size: int,
) -> pd.DataFrame:
    queries_path = retrieval_unit_dir(config, chunk_size) / "queries.jsonl"
    if not queries_path.exists():
        raise FileNotFoundError(
            f"{queries_path} does not exist. "
            f"Run `sclc chunk --chunk-size {chunk_size}` first."
        )

    queries = list(read_prepared_queries(queries_path))
    paragraph_maps, parent_headings = _build_paragraph_parent_map(
        config, chunk_size=chunk_size
    )
    query_types = load_query_types(config, {query.query_id for query in queries})

    rows: list[dict[str, Any]] = []
    for query in queries:
        paragraph_map = paragraph_maps.get(query.document_id)
        if paragraph_map is None:
            raise RuntimeError(f"Document {query.document_id} is missing from documents.jsonl")
        classification = classify_evidence_structure(query, paragraph_map)
        primary_parent_ids = {
            paragraph_map[paragraph_id]
            for evidence_set in query.evidence_sets
            if evidence_set.evidence_set_id == classification["primary_evidence_set_id"]
            for paragraph_id in evidence_set.paragraph_ids
        }
        rows.append(
            {
                "query_id": query.query_id,
                "document_id": query.document_id,
                "split": query.split,
                "analysis_set": query.analysis_set,
                "query_type": query_types.get(
                    query.query_id, query.query_type or "unclassified"
                ),
                "question": query.question,
                "primary_top_level_section_ids": "|".join(sorted(primary_parent_ids)),
                "primary_top_level_section_headings": "|".join(
                    parent_headings[parent_id]
                    for parent_id in sorted(primary_parent_ids)
                ),
                **classification,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No prepared queries are available for evidence-structure analysis")
    return frame.sort_values(["split", "analysis_set", "document_id", "query_id"])


def _evaluation_directory(
    config: AppConfig,
    condition: RetrievalCondition,
    model: EmbeddingModel | None,
    *,
    chunk_size: int,
) -> Path:
    directory = evaluation_dir(config, chunk_size) / condition.value
    if condition is not RetrievalCondition.BM25:
        if model is None:
            raise ValueError(f"Dense condition {condition.value} requires a model")
        directory = directory / model.value
    return directory


def _load_evaluation(
    config: AppConfig,
    condition: RetrievalCondition,
    model: EmbeddingModel | None,
    *,
    chunk_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = _evaluation_directory(
        config, condition, model, chunk_size=chunk_size
    )
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        label = condition.value if model is None else f"{condition.value}/{model.value}"
        raise FileNotFoundError(
            f"Evaluation for {label} is missing. Run `sclc evaluate` first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics_path = directory / manifest["files"]["query_metrics"]
    frame = pd.read_csv(
        metrics_path,
        dtype={"query_id": "string", "document_id": "string"},
    )
    return frame, manifest


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


def _structure_slices(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    slices = [("all", frame)]
    for structure in EVIDENCE_STRUCTURES:
        group = frame[frame["evidence_structure"] == structure]
        if not group.empty:
            slices.append((structure, group))
    return slices


def _summary_rows(
    frame: pd.DataFrame,
    *,
    model: EmbeddingModel,
    condition: RetrievalCondition,
    analysis_group: str,
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for structure, group in _structure_slices(frame):
        for metric in metrics:
            rows.append(
                {
                    "model_key": model.value,
                    "analysis_set": analysis_group,
                    "evidence_structure": structure,
                    "condition": condition.value,
                    "metric": metric,
                    "query_count": len(group),
                    "document_count": int(group["document_id"].nunique()),
                    "mean": float(group[metric].mean()),
                    "std": float(group[metric].std(ddof=1)) if len(group) > 1 else np.nan,
                    "minimum": float(group[metric].min()),
                    "maximum": float(group[metric].max()),
                }
            )
    return rows


def analyse_evidence_structure(
    config: AppConfig,
    *,
    chunk_size: int,
    models: Sequence[EmbeddingModel] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    selected_models = tuple(
        dict.fromkeys(tuple(EmbeddingModel) if models is None else models)
    )
    if not selected_models:
        raise ValueError("At least one embedding model must be selected")

    output_dir = analysis_dir(config, chunk_size) / "evidence_structure"
    if set(selected_models) != set(EmbeddingModel):
        output_dir = output_dir / "__".join(model.value for model in selected_models)
    query_path = output_dir / "evidence_structure_queries.csv"
    counts_path = output_dir / "evidence_structure_counts.csv"
    crosstab_path = output_dir / "evidence_structure_query_type_crosstab.csv"
    summary_path = output_dir / "summary_by_evidence_structure.csv"
    comparisons_path = output_dir / "scope_comparisons_by_evidence_structure.csv"
    manifest_path = output_dir / "manifest.json"
    bootstrap_dir = output_dir / "bootstrap"

    structure_frame = build_evidence_structure_frame(config, chunk_size=chunk_size)
    confirmatory_structure = structure_frame[
        structure_frame["split"] == config.evaluation.confirmatory_split
    ].copy()
    if confirmatory_structure.empty:
        raise RuntimeError(
            "No queries are available on the configured confirmatory split"
        )

    metric_columns = list(config.evaluation.bootstrap_metrics)
    source_manifests: dict[str, str] = {}
    loaded: dict[tuple[RetrievalCondition, EmbeddingModel | None], pd.DataFrame] = {}

    bm25, bm25_manifest = _load_evaluation(
        config,
        RetrievalCondition.BM25,
        None,
        chunk_size=chunk_size,
    )
    loaded[(RetrievalCondition.BM25, None)] = bm25
    source_manifests["bm25"] = bm25_manifest["configuration_fingerprint"]

    for model in selected_models:
        for condition in SUMMARY_CONDITIONS:
            if condition is RetrievalCondition.BM25:
                continue
            frame, manifest = _load_evaluation(
                config,
                condition,
                model,
                chunk_size=chunk_size,
            )
            loaded[(condition, model)] = frame
            source_manifests[f"{condition.value}/{model.value}"] = manifest[
                "configuration_fingerprint"
            ]

    for key, frame in loaded.items():
        missing = set(metric_columns).difference(frame.columns)
        if missing:
            condition, model = key
            label = condition.value if model is None else f"{condition.value}/{model.value}"
            raise ValueError(
                f"Evaluation {label} is missing metrics: {sorted(missing)}. "
                "Re-run `sclc evaluate --overwrite` if the metric configuration changed."
            )

    queries_path = retrieval_unit_dir(config, chunk_size) / "queries.jsonl"
    top_sections_path = retrieval_unit_dir(config, chunk_size) / "top_level_sections.jsonl"
    documents_path = config.paths.processed_dir / "documents.jsonl"
    query_types_path = config.paths.retrieval_unit_dir / config.evaluation.query_types_filename
    configuration = {
        "chunk_size_tokens": chunk_size,
        "models": [model.value for model in selected_models],
        "confirmatory_split": config.evaluation.confirmatory_split,
        "metrics": metric_columns,
        "comparisons": [
            [first.value, second.value] for first, second in SCOPE_COMPARISONS
        ],
        "bootstrap_iterations": config.evaluation.bootstrap_iterations,
        "confidence_level": config.evaluation.confidence_level,
        "seed": config.project.seed,
        "classification_policy": {
            "unit": "top_level_academic_section",
            "primary_set": "minimum_paragraph_count_complete_acceptable_set",
            "tie_break": "least_distributed_structure",
            "tie_is_reported": True,
            "structures": list(EVIDENCE_STRUCTURES),
        },
        "source_hashes": {
            "queries": _file_sha256(queries_path),
            "top_level_sections": _file_sha256(top_sections_path),
            "documents": _file_sha256(documents_path),
            "query_types": _file_sha256(query_types_path),
        },
        "source_manifests": source_manifests,
    }
    fingerprint = _fingerprint(configuration)
    if manifest_path.exists() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("configuration_fingerprint") != fingerprint:
            raise RuntimeError(
                f"Cached evidence-structure analysis at {output_dir} does not match "
                "current inputs. Re-run with --overwrite."
            )
        return existing

    output_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in bootstrap_dir.glob("*.npz"):
            path.unlink()

    structure_frame.to_csv(query_path, index=False)

    counts = (
        structure_frame.groupby(
            ["split", "analysis_set", "evidence_structure"],
            dropna=False,
            as_index=False,
        )
        .agg(
            query_count=("query_id", "count"),
            document_count=("document_id", "nunique"),
            ambiguous_minimal_structure_count=(
                "minimal_structure_ambiguous",
                "sum",
            ),
            cross_section_required_count=("cross_section_required", "sum"),
            cross_section_required_among_minimal_sets_count=(
                "cross_section_required_among_minimal_sets",
                "sum",
            ),
            cross_section_possible_count=("cross_section_possible", "sum"),
            mean_minimum_evidence_paragraph_count=(
                "minimum_evidence_paragraph_count",
                "mean",
            ),
        )
        .sort_values(["split", "analysis_set", "evidence_structure"])
    )
    counts.to_csv(counts_path, index=False)

    crosstab = (
        structure_frame.groupby(
            ["split", "analysis_set", "evidence_structure", "query_type"],
            dropna=False,
            as_index=False,
        )
        .agg(
            query_count=("query_id", "count"),
            document_count=("document_id", "nunique"),
        )
        .sort_values(
            ["split", "analysis_set", "evidence_structure", "query_type"]
        )
    )
    crosstab.to_csv(crosstab_path, index=False)

    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    seed_offset = 0

    structure_metadata = confirmatory_structure[
        [
            "query_id",
            "document_id",
            "analysis_set",
            "query_type",
            "question",
            "evidence_structure",
            "minimum_evidence_paragraph_count",
            "minimal_structure_ambiguous",
            "cross_section_required",
            "cross_section_required_among_minimal_sets",
        ]
    ]

    for model in selected_models:
        eligible_sets = (
            {"cross_model_core"}
            if model is EmbeddingModel.JINA
            else {"cross_model_core", "granite_extended"}
        )
        expected_structure = structure_metadata[
            structure_metadata["analysis_set"].isin(eligible_sets)
        ].copy()
        model_loaded: dict[RetrievalCondition, pd.DataFrame] = {}
        for condition in SUMMARY_CONDITIONS:
            key = (
                (RetrievalCondition.BM25, None)
                if condition is RetrievalCondition.BM25
                else (condition, model)
            )
            raw = loaded[key]
            raw = raw[
                (raw["split"] == config.evaluation.confirmatory_split)
                & raw["analysis_set"].isin(eligible_sets)
            ].copy()
            selected = raw[["query_id", "document_id", *metric_columns]].merge(
                expected_structure,
                on=["query_id", "document_id"],
                how="inner",
                validate="one_to_one",
            )
            if len(selected) != len(expected_structure):
                raise RuntimeError(
                    f"Evidence-structure metadata and {condition.value}/{model.value} "
                    "evaluation query sets differ"
                )
            model_loaded[condition] = selected

        for analysis_name, analysis_sets in _analysis_groups(model):
            for condition in SUMMARY_CONDITIONS:
                group = model_loaded[condition][
                    model_loaded[condition]["analysis_set"].isin(analysis_sets)
                ]
                if group.empty:
                    continue
                summary_rows.extend(
                    _summary_rows(
                        group,
                        model=model,
                        condition=condition,
                        analysis_group=analysis_name,
                        metrics=metric_columns,
                    )
                )

            for first, second in SCOPE_COMPARISONS:
                first_frame = model_loaded[first][
                    model_loaded[first]["analysis_set"].isin(analysis_sets)
                ]
                second_frame = model_loaded[second][
                    model_loaded[second]["analysis_set"].isin(analysis_sets)
                ]
                if first_frame.empty or second_frame.empty:
                    continue
                metadata_columns = [
                    "query_id",
                    "document_id",
                    "analysis_set",
                    "query_type",
                    "question",
                    "evidence_structure",
                    "minimum_evidence_paragraph_count",
                    "minimal_structure_ambiguous",
                    "cross_section_required",
                ]
                first_selected = first_frame[metadata_columns + metric_columns].rename(
                    columns={metric: f"first__{metric}" for metric in metric_columns}
                )
                second_selected = second_frame[
                    metadata_columns + metric_columns
                ].rename(
                    columns={metric: f"second__{metric}" for metric in metric_columns}
                )
                merged = first_selected.merge(
                    second_selected,
                    on=metadata_columns,
                    how="inner",
                    validate="one_to_one",
                )
                if len(merged) != len(first_frame) or len(merged) != len(second_frame):
                    raise RuntimeError(
                        f"Query sets differ for {first.value} vs {second.value}, "
                        f"{model.value}/{analysis_name}"
                    )

                for structure, group in _structure_slices(merged):
                    for metric in metric_columns:
                        seed_offset += 1
                        observed, lower, upper, p_value, samples = (
                            paired_document_bootstrap(
                                group,
                                first_column=f"first__{metric}",
                                second_column=f"second__{metric}",
                                iterations=config.evaluation.bootstrap_iterations,
                                confidence_level=config.evaluation.confidence_level,
                                seed=config.project.seed + seed_offset,
                            )
                        )
                        comparison_id = (
                            f"{model.value}__{analysis_name}__{structure}__"
                            f"{first.value}__{second.value}__{metric}"
                        )
                        safe_name = hashlib.sha1(
                            comparison_id.encode("utf-8"), usedforsecurity=False
                        ).hexdigest()[:16]
                        bootstrap_file = bootstrap_dir / f"bootstrap_{safe_name}.npz"
                        np.savez_compressed(
                            bootstrap_file,
                            differences=samples,
                            observed_difference=np.asarray(
                                [observed], dtype=np.float64
                            ),
                        )
                        comparison_rows.append(
                            {
                                "model_key": model.value,
                                "analysis_set": analysis_name,
                                "evidence_structure": structure,
                                "first_condition": first.value,
                                "second_condition": second.value,
                                "metric": metric,
                                "query_count": len(group),
                                "document_count": int(group["document_id"].nunique()),
                                "low_sample_warning": bool(
                                    len(group) < 20
                                    or group["document_id"].nunique() < 10
                                ),
                                "ambiguous_minimal_structure_count": int(
                                    group["minimal_structure_ambiguous"].sum()
                                ),
                                "first_mean": float(group[f"first__{metric}"].mean()),
                                "second_mean": float(
                                    group[f"second__{metric}"].mean()
                                ),
                                "mean_difference_second_minus_first": observed,
                                "confidence_lower": lower,
                                "confidence_upper": upper,
                                "bootstrap_p_value": p_value,
                                "significant_by_ci": bool(
                                    lower > 0.0 or upper < 0.0
                                ),
                                "bootstrap_file": str(
                                    bootstrap_file.relative_to(output_dir)
                                ),
                            }
                        )

    summary = pd.DataFrame(summary_rows).sort_values(
        [
            "model_key",
            "analysis_set",
            "evidence_structure",
            "condition",
            "metric",
        ]
    )
    summary.to_csv(summary_path, index=False)

    comparisons = pd.DataFrame(comparison_rows)
    if comparisons.empty:
        raise RuntimeError("No evidence-structure comparisons were generated")
    comparisons["holm_adjusted_p_value"] = np.nan
    family_columns = [
        "model_key",
        "analysis_set",
        "evidence_structure",
        "metric",
    ]
    for _, indices in comparisons.groupby(family_columns, dropna=False).groups.items():
        index_list = list(indices)
        adjusted = holm_adjust(
            comparisons.loc[index_list, "bootstrap_p_value"].tolist()
        )
        comparisons.loc[index_list, "holm_adjusted_p_value"] = adjusted
    comparisons["significant_after_holm_0_05"] = (
        comparisons["holm_adjusted_p_value"] < 0.05
    )
    comparisons = comparisons.sort_values(
        [
            "model_key",
            "analysis_set",
            "evidence_structure",
            "first_condition",
            "second_condition",
            "metric",
        ]
    )
    comparisons.to_csv(comparisons_path, index=False)

    test_counts = (
        confirmatory_structure.groupby("evidence_structure")["query_id"]
        .count()
        .to_dict()
    )
    manifest = {
        "schema_version": 1,
        "kind": "evidence_structure_analysis",
        "chunk_size_tokens": chunk_size,
        "query_count": len(structure_frame),
        "confirmatory_query_count": len(confirmatory_structure),
        "confirmatory_structure_counts": {
            structure: int(test_counts.get(structure, 0))
            for structure in EVIDENCE_STRUCTURES
        },
        "summary_row_count": len(summary),
        "comparison_count": len(comparisons),
        "configuration": configuration,
        "configuration_fingerprint": fingerprint,
        "files": {
            "query_classifications": query_path.name,
            "structure_counts": counts_path.name,
            "query_type_crosstab": crosstab_path.name,
            "summary": summary_path.name,
            "scope_comparisons": comparisons_path.name,
            "bootstrap_directory": bootstrap_dir.name,
        },
        "interpretation_notes": [
            "Evidence structure describes gold-support distribution, not reasoning depth.",
            "The original held-out test distribution is unchanged.",
            "Low-count strata are secondary exploratory analyses.",
            "A cross-section label is assigned only when the least-distributed "
            "minimal acceptable evidence route still crosses top-level sections.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


__all__ = [
    "EVIDENCE_STRUCTURES",
    "MULTI_PARAGRAPH_CROSS_SECTION",
    "MULTI_PARAGRAPH_SAME_SECTION",
    "SCOPE_COMPARISONS",
    "SINGLE_PARAGRAPH",
    "analyse_evidence_structure",
    "build_evidence_structure_frame",
    "classify_evidence_structure",
]
