import json
from pathlib import Path

import pandas as pd
import pytest

from sclc.analysis.scope_effect import analyse_scope_effect
from sclc.paths import analysis_dir, evaluation_dir, retrieval_unit_dir
from sclc.options import EmbeddingModel
from sclc.config import AppConfig
from sclc.data.retrieval_unit_io import write_models_jsonl
from sclc.data.schema import CharacterSpan, TopLevelSectionRecord


def make_config(tmp_path: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "project": {"seed": 42},
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
            "chunking": {"canonical_tokenizer": "granite"},
            "models": {
                "granite": {"model_id": "granite", "max_document_tokens": 32768},
                "jina": {"model_id": "jina", "max_document_tokens": 8192},
            },
            "sampling": {},
            "evaluation": {"primary_metric": "ndcg_at_10"},
        }
    )


def test_scope_effect_analysis_aggregates_within_document(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.paths.subset_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "document_id": "d1",
                "analysis_set": "cross_model_core",
                "granite_tokens": 1000,
                "jina_tokens": 900,
                "character_count": 5000,
            },
            {
                "document_id": "d2",
                "analysis_set": "cross_model_core",
                "granite_tokens": 2000,
                "jina_tokens": 1800,
                "character_count": 9000,
            },
            {
                "document_id": "d3",
                "analysis_set": "granite_extended",
                "granite_tokens": 10000,
                "jina_tokens": 9000,
                "character_count": 40000,
            },
        ]
    ).to_csv(config.paths.subset_dir / "selected_documents.csv", index=False)
    sections = []
    for document_id, analysis_set, sizes in (
        ("d1", "cross_model_core", [100, 200]),
        ("d2", "cross_model_core", [200, 300]),
        ("d3", "granite_extended", [500, 700, 800]),
    ):
        cursor = 0
        for index, size in enumerate(sizes):
            sections.append(
                TopLevelSectionRecord(
                    parent_section_id=f"{document_id}:s{index}",
                    document_id=document_id,
                    analysis_set=analysis_set,
                    heading=f"S{index}",
                    span=CharacterSpan(start=cursor, end=cursor + size),
                    source_section_ids=[],
                    text="x" * size,
                )
            )
            cursor += size
    write_models_jsonl(
        sections, retrieval_unit_dir(config, 512) / "top_level_sections.jsonl"
    )

    def write_eval(condition: str, model: str, rows: list[dict[str, object]]) -> None:
        directory = evaluation_dir(config, 512) / condition / model
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(directory / "query_metrics.csv", index=False)
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "configuration_fingerprint": f"{condition}-{model}",
                    "files": {"query_metrics": "query_metrics.csv"},
                }
            ),
            encoding="utf-8",
        )

    base = [
        {
            "query_id": "q1",
            "document_id": "d1",
            "analysis_set": "cross_model_core",
            "split": "test",
            "query_type": "factual",
            "question": "Q1",
        },
        {
            "query_id": "q2",
            "document_id": "d1",
            "analysis_set": "cross_model_core",
            "split": "test",
            "query_type": "factual",
            "question": "Q2",
        },
        {
            "query_id": "q3",
            "document_id": "d2",
            "analysis_set": "cross_model_core",
            "split": "test",
            "query_type": "synthesis",
            "question": "Q3",
        },
    ]
    extended = {
        "query_id": "q4",
        "document_id": "d3",
        "analysis_set": "granite_extended",
        "split": "test",
        "query_type": "multi_hop",
        "question": "Q4",
    }
    for model in ("granite", "jina"):
        model_rows = base + ([extended] if model == "granite" else [])
        section_rows = [dict(row, ndcg_at_10=0.6) for row in model_rows]
        global_rows = [dict(row, ndcg_at_10=0.4) for row in model_rows]
        write_eval("section_constrained", model, section_rows)
        write_eval("global", model, global_rows)

    manifest = analyse_scope_effect(config, chunk_size=512)
    output_dir = analysis_dir(config, 512) / "scope_effect"
    documents = pd.read_csv(output_dir / manifest["files"]["document_effects"])
    d1 = documents[(documents["model_key"] == "granite") & (documents["document_id"] == "d1")]
    assert d1.iloc[0]["query_count"] == 2
    assert d1.iloc[0]["mean_scope_effect_section_minus_global"] == pytest.approx(0.2)
    associations = pd.read_csv(output_dir / manifest["files"]["associations"])
    assert set(associations["analysis_set"]).issuperset(
        {"cross_model_core", "granite_extended", "all_eligible"}
    )

    # Granite-only analysis must remain runnable before Jina robustness outputs exist.
    import shutil

    shutil.rmtree(evaluation_dir(config, 512) / "section_constrained" / "jina")
    shutil.rmtree(evaluation_dir(config, 512) / "global" / "jina")
    granite_manifest = analyse_scope_effect(
        config, chunk_size=512, models=(EmbeddingModel.GRANITE,)
    )
    granite_output_dir = analysis_dir(config, 512) / "scope_effect" / "granite"
    granite_documents = pd.read_csv(
        granite_output_dir / granite_manifest["files"]["document_effects"]
    )
    assert set(granite_documents["model_key"]) == {"granite"}
