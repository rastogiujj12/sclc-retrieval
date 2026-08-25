import json
from pathlib import Path

import pandas as pd

from sclc.analysis.evidence_structure import (
    MULTI_PARAGRAPH_CROSS_SECTION,
    MULTI_PARAGRAPH_SAME_SECTION,
    SINGLE_PARAGRAPH,
    analyse_evidence_structure,
    classify_evidence_structure,
)
from sclc.config import AppConfig
from sclc.data.io import write_documents_jsonl
from sclc.data.retrieval_unit_io import write_models_jsonl
from sclc.data.schema import (
    CharacterSpan,
    DocumentRecord,
    EvidenceSetRecord,
    ParagraphRecord,
    PreparedQueryRecord,
    SectionRecord,
    TopLevelSectionRecord,
)
from sclc.options import EmbeddingModel
from sclc.paths import analysis_dir, evaluation_dir, retrieval_unit_dir


def make_query(
    query_id: str,
    evidence_sets: list[list[str]],
    *,
    document_id: str = "d1",
    analysis_set: str = "cross_model_core",
) -> PreparedQueryRecord:
    return PreparedQueryRecord(
        query_id=query_id,
        document_id=document_id,
        split="test",
        analysis_set=analysis_set,
        question=f"Question {query_id}",
        evidence_union_paragraph_ids=sorted(
            {paragraph for evidence_set in evidence_sets for paragraph in evidence_set}
        ),
        evidence_sets=[
            EvidenceSetRecord(
                evidence_set_id=f"{query_id}:set:{index}",
                paragraph_ids=evidence_set,
            )
            for index, evidence_set in enumerate(evidence_sets)
        ],
    )


def test_classification_uses_minimal_complete_acceptable_set() -> None:
    paragraph_to_parent = {"p1": "s1", "p2": "s1", "p3": "s2"}

    single = classify_evidence_structure(
        make_query("q1", [["p1"], ["p2", "p3"]]),
        paragraph_to_parent,
    )
    assert single["evidence_structure"] == SINGLE_PARAGRAPH
    assert single["cross_section_possible"] is True
    assert single["cross_section_required"] is False

    same = classify_evidence_structure(
        make_query("q2", [["p1", "p2"]]),
        paragraph_to_parent,
    )
    assert same["evidence_structure"] == MULTI_PARAGRAPH_SAME_SECTION

    cross = classify_evidence_structure(
        make_query("q3", [["p1", "p3"]]),
        paragraph_to_parent,
    )
    assert cross["evidence_structure"] == MULTI_PARAGRAPH_CROSS_SECTION
    assert cross["cross_section_required"] is True
    assert cross["cross_section_required_among_minimal_sets"] is True

    ambiguous = classify_evidence_structure(
        make_query("q4", [["p1", "p2"], ["p1", "p3"]]),
        paragraph_to_parent,
    )
    assert ambiguous["evidence_structure"] == MULTI_PARAGRAPH_SAME_SECTION
    assert ambiguous["minimal_structure_ambiguous"] is True
    assert (
        ambiguous["minimal_structure_variants"]
        == "multi_paragraph_same_section|multi_paragraph_cross_section"
    )


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
            "chunking": {
                "canonical_tokenizer": "granite",
                "chunk_size_tokens": 128,
            },
            "models": {
                "granite": {"model_id": "granite", "max_document_tokens": 32768},
                "jina": {"model_id": "jina", "max_document_tokens": 8192},
            },
            "sampling": {},
            "evaluation": {
                "bootstrap_iterations": 50,
                "bootstrap_metrics": ["ndcg_at_5"],
                "primary_metric": "ndcg_at_5",
            },
        }
    )


def make_document(document_id: str) -> DocumentRecord:
    text = "A" * 100
    sections = [
        SectionRecord(
            section_id=f"{document_id}:section:1",
            source_section_index=0,
            heading="Methods",
            top_level_heading="Methods",
            hierarchy=["Methods"],
            span=CharacterSpan(start=0, end=50),
            paragraph_ids=[f"{document_id}:p1", f"{document_id}:p2"],
        ),
        SectionRecord(
            section_id=f"{document_id}:section:2",
            source_section_index=1,
            heading="Results",
            top_level_heading="Results",
            hierarchy=["Results"],
            span=CharacterSpan(start=50, end=100),
            paragraph_ids=[f"{document_id}:p3"],
        ),
    ]
    paragraphs = [
        ParagraphRecord(
            paragraph_id=f"{document_id}:p1",
            section_id=f"{document_id}:section:1",
            source_section_index=0,
            source_paragraph_index=0,
            text="A" * 20,
            span=CharacterSpan(start=0, end=20),
        ),
        ParagraphRecord(
            paragraph_id=f"{document_id}:p2",
            section_id=f"{document_id}:section:1",
            source_section_index=0,
            source_paragraph_index=1,
            text="A" * 20,
            span=CharacterSpan(start=25, end=45),
        ),
        ParagraphRecord(
            paragraph_id=f"{document_id}:p3",
            section_id=f"{document_id}:section:2",
            source_section_index=1,
            source_paragraph_index=0,
            text="A" * 20,
            span=CharacterSpan(start=55, end=75),
        ),
    ]
    return DocumentRecord(
        document_id=document_id,
        split="test",
        title=document_id,
        abstract="",
        text=text,
        sections=sections,
        paragraphs=paragraphs,
        queries=[],
    )


def test_analysis_writes_stratified_summaries_and_bootstrap_outputs(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    config.paths.processed_dir.mkdir(parents=True)
    write_documents_jsonl(
        [make_document("d1"), make_document("d2")],
        config.paths.processed_dir / "documents.jsonl",
    )

    retrieval_dir = retrieval_unit_dir(config, 128)
    retrieval_dir.mkdir(parents=True)
    queries = [
        make_query("q1", [["d1:p1"]]),
        make_query("q2", [["d1:p1", "d1:p2"]]),
        make_query("q3", [["d1:p1", "d1:p3"]]),
        make_query(
            "q4",
            [["d2:p1"]],
            document_id="d2",
            analysis_set="granite_extended",
        ),
    ]
    write_models_jsonl(queries, retrieval_dir / "queries.jsonl")

    parents = []
    for document_id, analysis_set in (
        ("d1", "cross_model_core"),
        ("d2", "granite_extended"),
    ):
        parents.extend(
            [
                TopLevelSectionRecord(
                    parent_section_id=f"{document_id}:parent:1",
                    document_id=document_id,
                    analysis_set=analysis_set,
                    heading="Methods",
                    span=CharacterSpan(start=0, end=50),
                    source_section_ids=[f"{document_id}:section:1"],
                    text="A" * 50,
                ),
                TopLevelSectionRecord(
                    parent_section_id=f"{document_id}:parent:2",
                    document_id=document_id,
                    analysis_set=analysis_set,
                    heading="Results",
                    span=CharacterSpan(start=50, end=100),
                    source_section_ids=[f"{document_id}:section:2"],
                    text="A" * 50,
                ),
            ]
        )
    write_models_jsonl(parents, retrieval_dir / "top_level_sections.jsonl")

    config.paths.retrieval_unit_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "query_id": ["q1", "q2", "q3", "q4"],
            "query_type": ["factual", "synthesis", "multi_hop", "factual"],
        }
    ).to_csv(config.paths.retrieval_unit_dir / "query_types.csv", index=False)

    scores = {
        "bm25": 0.1,
        "fixed_dense": 0.2,
        "section_isolated": 0.5,
        "section_constrained": 0.4,
        "global": 0.3,
    }

    def write_evaluation(condition: str, model: str | None) -> None:
        directory = evaluation_dir(config, 128) / condition
        if model is not None:
            directory = directory / model
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        for query in queries:
            rows.append(
                {
                    "query_id": query.query_id,
                    "document_id": query.document_id,
                    "analysis_set": query.analysis_set,
                    "split": "test",
                    "ndcg_at_5": scores[condition],
                }
            )
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

    write_evaluation("bm25", None)
    for condition in (
        "fixed_dense",
        "section_isolated",
        "section_constrained",
        "global",
    ):
        write_evaluation(condition, "granite")

    manifest = analyse_evidence_structure(
        config,
        chunk_size=128,
        models=(EmbeddingModel.GRANITE,),
    )
    output_dir = analysis_dir(config, 128) / "evidence_structure" / "granite"
    queries_frame = pd.read_csv(
        output_dir / manifest["files"]["query_classifications"]
    )
    assert set(queries_frame["evidence_structure"]) == {
        SINGLE_PARAGRAPH,
        MULTI_PARAGRAPH_SAME_SECTION,
        MULTI_PARAGRAPH_CROSS_SECTION,
    }

    comparisons = pd.read_csv(
        output_dir / manifest["files"]["scope_comparisons"]
    )
    assert set(comparisons["evidence_structure"]).issuperset(
        {
            "all",
            SINGLE_PARAGRAPH,
            MULTI_PARAGRAPH_SAME_SECTION,
            MULTI_PARAGRAPH_CROSS_SECTION,
        }
    )
    assert set(comparisons["first_condition"]) == {
        "section_isolated",
        "section_constrained",
    }
    assert list((output_dir / "bootstrap").glob("*.npz"))
