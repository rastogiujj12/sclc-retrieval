from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sclc.analysis.chunk_size_pilot import analyse_chunk_size_pilot
from sclc.config import AppConfig
from sclc.paths import evaluation_dir


def make_config(tmp_path: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "project": {"seed": 9},
            "paths": {
                "raw_dir": tmp_path / "raw",
                "processed_dir": tmp_path / "processed",
                "profile_dir": tmp_path / "profiles",
                "subset_dir": tmp_path / "subsets",
                "retrieval_unit_dir": tmp_path / "retrieval_units",
                "encoding_dir": tmp_path / "encodings",
                "ranking_dir": tmp_path / "rankings",
                "evaluation_dir": tmp_path / "evaluation",
                "analysis_dir": tmp_path / "analysis",
                "hf_cache_dir": tmp_path / "cache",
            },
            "dataset": {"repo_id": "allenai/qasper"},
            "document": {},
            "chunking": {
                "canonical_tokenizer": "granite",
                "supported_chunk_sizes": [128, 256, 512],
            },
            "models": {
                "granite": {"model_id": "granite", "max_document_tokens": 32768},
                "jina": {"model_id": "jina", "max_document_tokens": 8192},
            },
            "sampling": {},
            "evaluation": {"bootstrap_iterations": 30},
            "pilot": {
                "chunk_sizes": [128, 256, 512],
                "selection_split": "validation",
                "practical_equivalence_margin": 0.01,
            },
        }
    )


def write_metrics(
    config: AppConfig,
    *,
    chunk_size: int,
    condition: str,
    model: str | None,
    primary: float,
    secondary: float,
) -> None:
    directory = evaluation_dir(config, chunk_size) / condition
    if model is not None:
        directory = directory / model
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for query_id, document_id in (("q1", "d1"), ("q2", "d2")):
        rows.append(
            {
                "query_id": query_id,
                "document_id": document_id,
                "split": "validation",
                "candidate_count": 20,
                "cutoff_saturated_at_5": 0.0,
                "cutoff_saturated_at_10": 0.0,
                "evidence_paragraph_recall_at_token_budget_1024": primary,
                "complete_evidence_at_token_budget_2048": secondary,
                "average_precision": primary - 0.05,
                "ndcg_at_5": primary - 0.02,
                "recall_at_5": primary - 0.03,
            }
        )
    # Test data deliberately favours 128, but it must not affect selection.
    rows.append(
        {
            "query_id": "q-test",
            "document_id": "d-test",
            "split": "test",
            "candidate_count": 20,
            "cutoff_saturated_at_5": 0.0,
            "cutoff_saturated_at_10": 0.0,
            "evidence_paragraph_recall_at_token_budget_1024": (
                1.0 if chunk_size == 128 else 0.0
            ),
            "complete_evidence_at_token_budget_2048": (
                1.0 if chunk_size == 128 else 0.0
            ),
            "average_precision": 1.0 if chunk_size == 128 else 0.0,
            "ndcg_at_5": 1.0 if chunk_size == 128 else 0.0,
            "recall_at_5": 1.0 if chunk_size == 128 else 0.0,
        }
    )
    pd.DataFrame(rows).to_csv(directory / "query_metrics.csv", index=False)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "configuration_fingerprint": f"{condition}-{model}-{chunk_size}",
                "files": {"query_metrics": "query_metrics.csv"},
            }
        ),
        encoding="utf-8",
    )


def test_pilot_selects_from_validation_only_and_uses_efficiency_tiebreak(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    fixed_scores = {
        128: (0.700, 0.700),
        256: (0.705, 0.720),
        512: (0.710, 0.710),
    }
    for size in (128, 256, 512):
        write_metrics(
            config,
            chunk_size=size,
            condition="bm25",
            model=None,
            primary=0.60 + size / 10000,
            secondary=0.65,
        )
        primary, secondary = fixed_scores[size]
        write_metrics(
            config,
            chunk_size=size,
            condition="fixed_dense",
            model="granite",
            primary=primary,
            secondary=secondary,
        )

    manifest = analyse_chunk_size_pilot(config)
    assert manifest["selected_chunk_size_tokens"] == 512
    selection = json.loads(
        (config.paths.analysis_dir / "chunk_size_pilot" / "selection.json").read_text()
    )
    assert selection["test_split_used_for_selection"] is False
    assert selection["details"]["secondary_eligible_chunk_sizes"] == [256, 512]
