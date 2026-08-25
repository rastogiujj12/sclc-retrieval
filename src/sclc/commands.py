from __future__ import annotations

from pathlib import Path

from rich.console import Console
from tqdm import tqdm

from sclc.analysis.chunk_size_pilot import analyse_chunk_size_pilot
from sclc.analysis.chunk_size_sensitivity import analyse_chunk_size_sensitivity
from sclc.analysis.cross_section_challenge import freeze_cross_section_challenge
from sclc.analysis.evidence_structure import analyse_evidence_structure
from sclc.analysis.qasper_collection import audit_qasper_collection
from sclc.analysis.scope_effect import analyse_scope_effect
from sclc.config import ensure_output_directories, load_config
from sclc.data.chunk_pipeline import construct_retrieval_units
from sclc.data.io import (
    document_summary_frame,
    evidence_alignment_frame,
    write_documents_jsonl,
)
from sclc.data.profile import profile_documents, write_profile_outputs
from sclc.data.qasper import iter_raw_records
from sclc.data.query_types import validate_query_type_coding
from sclc.data.reconstruct import reconstruct_document
from sclc.data.sample import read_profile_csv, select_documents
from sclc.encoding.pipeline import encode_condition
from sclc.evaluation.metrics import evaluate_condition
from sclc.evaluation.statistics import compare_conditions
from sclc.options import EmbeddingModel, RetrievalCondition
from sclc.paths import (
    analysis_dir,
    encoding_dir,
    evaluation_dir,
    ranking_dir,
    resolve_chunk_size,
    retrieval_unit_dir,
)
from sclc.retrieval.ranking import rank_condition

console = Console()


def prepare_command(config_path: Path) -> None:
    config = load_config(config_path)
    ensure_output_directories(config)

    documents = []
    for split, raw in tqdm(iter_raw_records(config), desc="Preparing QASPER papers"):
        documents.append(reconstruct_document(raw, split=split, config=config.document))

    documents_path = config.paths.processed_dir / "documents.jsonl"
    write_documents_jsonl(documents, documents_path)

    document_summary_frame(documents).to_csv(
        config.paths.processed_dir / "document_summary.csv",
        index=False,
    )
    evidence_alignment_frame(documents).to_csv(
        config.paths.processed_dir / "evidence_alignment.csv",
        index=False,
    )

    console.print(f"[green]Prepared {len(documents)} papers.[/green]")
    console.print(f"Documents: {documents_path}")


def profile_command(config_path: Path) -> None:
    config = load_config(config_path)
    ensure_output_directories(config)
    frame = profile_documents(config)
    write_profile_outputs(frame, config.paths.profile_dir)

    counts = frame["eligibility_group"].value_counts().to_dict()
    console.print("[green]Document profiling complete.[/green]")
    for group, count in counts.items():
        console.print(f"{group}: {count}")


def sample_command(config_path: Path) -> None:
    config = load_config(config_path)
    ensure_output_directories(config)

    profile_path = config.paths.profile_dir / "document_lengths.csv"
    if not profile_path.exists():
        raise FileNotFoundError(f"{profile_path} does not exist. Run `sclc profile` first.")

    profile = read_profile_csv(profile_path)
    selected = select_documents(config, profile)
    output_path = config.paths.subset_dir / "selected_documents.csv"
    selected.to_csv(output_path, index=False)

    console.print(f"[green]Selected {len(selected)} papers.[/green]")
    console.print(f"Subset manifest: {output_path}")


def chunk_command(config_path: Path, *, chunk_size: int | None) -> None:
    config = load_config(config_path)
    ensure_output_directories(config)
    resolved_chunk_size = resolve_chunk_size(config, chunk_size)
    validation = construct_retrieval_units(
        config, chunk_size_tokens=resolved_chunk_size
    )

    counts = validation["counts"]
    console.print("[green]Retrieval-unit construction complete.[/green]")
    console.print(f"Continuous units: {counts['continuous_units']}")
    console.print(f"Section-bounded units: {counts['section_bounded_units']}")
    console.print(f"Usable queries: {counts['usable_queries']}")
    console.print(f"Validation: {validation['status']}")
    console.print(f"Outputs: {retrieval_unit_dir(config, resolved_chunk_size)}")


def encode_command(
    config_path: Path,
    *,
    condition: RetrievalCondition,
    model: EmbeddingModel | None,
    chunk_size: int | None,
    overwrite: bool,
) -> None:
    config = load_config(config_path)
    ensure_output_directories(config)
    resolved_chunk_size = resolve_chunk_size(config, chunk_size)

    if condition.is_dense and model is not None:
        model_config = (
            config.models.granite
            if model is EmbeddingModel.GRANITE
            else config.models.jina
        )
        if model_config.revision is None:
            console.print(
                "[yellow]Warning: the model revision is not pinned. Set an immutable "
                "Hugging Face commit hash before the final dissertation run.[/yellow]"
            )

    try:
        result = encode_condition(
            config,
            condition=condition,
            model=model,
            chunk_size=resolved_chunk_size,
            overwrite=overwrite,
        )
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise

    console.print(
        f"[green]Encoding complete for {condition.value} at {resolved_chunk_size} tokens.[/green]"
    )
    console.print(f"Documents: {result['document_count']}")
    console.print(f"Retrieval units: {result['unit_count']}")
    if "query_count" in result:
        console.print(f"Queries: {result['query_count']}")
    if condition is RetrievalCondition.BM25:
        output_path = encoding_dir(config, resolved_chunk_size) / "bm25"
    else:
        assert model is not None
        output_path = (
            encoding_dir(config, resolved_chunk_size) / condition.value / model.value
        )
    console.print(f"Outputs: {output_path}")


def retrieve_command(
    config_path: Path,
    *,
    condition: RetrievalCondition,
    model: EmbeddingModel | None,
    chunk_size: int | None,
    overwrite: bool,
) -> None:
    config = load_config(config_path)
    ensure_output_directories(config)
    resolved_chunk_size = resolve_chunk_size(config, chunk_size)
    validate_query_type_coding(
        config, model, chunk_size=resolved_chunk_size
    )
    result = rank_condition(
        config,
        condition=condition,
        model=model,
        chunk_size=resolved_chunk_size,
        overwrite=overwrite,
    )
    console.print(
        f"[green]Retrieval complete for {condition.value} at "
        f"{resolved_chunk_size} tokens.[/green]"
    )
    console.print(f"Queries: {result['query_count']}")
    console.print(f"Ranked results: {result['result_count']}")
    output_path = (
        ranking_dir(config, resolved_chunk_size) / condition.value
        if condition is RetrievalCondition.BM25
        else ranking_dir(config, resolved_chunk_size) / condition.value / model.value
    )
    console.print(f"Outputs: {output_path}")


def evaluate_command(
    config_path: Path,
    *,
    condition: RetrievalCondition,
    model: EmbeddingModel | None,
    chunk_size: int | None,
    overwrite: bool,
) -> None:
    config = load_config(config_path)
    ensure_output_directories(config)
    resolved_chunk_size = resolve_chunk_size(config, chunk_size)
    result = evaluate_condition(
        config,
        condition=condition,
        model=model,
        chunk_size=resolved_chunk_size,
        overwrite=overwrite,
    )
    console.print(
        f"[green]Evaluation complete for {condition.value} at "
        f"{resolved_chunk_size} tokens.[/green]"
    )
    console.print(f"Queries: {result['query_count']}")
    console.print(f"Documents: {result['document_count']}")
    console.print(f"Classified queries: {result['classified_query_count']}")
    output_path = (
        evaluation_dir(config, resolved_chunk_size) / condition.value
        if condition is RetrievalCondition.BM25
        else evaluation_dir(config, resolved_chunk_size) / condition.value / model.value
    )
    console.print(f"Outputs: {output_path}")


def compare_command(
    config_path: Path,
    *,
    model: EmbeddingModel | None,
    chunk_size: int | None,
    overwrite: bool,
) -> None:
    config = load_config(config_path)
    ensure_output_directories(config)
    resolved_chunk_size = resolve_chunk_size(config, chunk_size)
    selected_models = None if model is None else (model,)
    result = compare_conditions(
        config,
        chunk_size=resolved_chunk_size,
        models=selected_models,
        overwrite=overwrite,
    )
    console.print("[green]Pairwise statistical comparisons complete.[/green]")
    console.print(f"Comparisons: {result['comparison_count']}")
    comparison_path = evaluation_dir(config, resolved_chunk_size) / "comparisons"
    if model is not None:
        comparison_path = comparison_path / model.value
    console.print(f"Outputs: {comparison_path}")


def analyse_command(
    config_path: Path,
    *,
    model: EmbeddingModel | None,
    chunk_size: int | None,
    overwrite: bool,
) -> None:
    config = load_config(config_path)
    ensure_output_directories(config)
    resolved_chunk_size = resolve_chunk_size(config, chunk_size)
    selected_models = None if model is None else (model,)
    result = analyse_scope_effect(
        config,
        chunk_size=resolved_chunk_size,
        models=selected_models,
        overwrite=overwrite,
    )
    console.print("[green]Scope-effect analysis complete.[/green]")
    console.print(f"Query effects: {result['query_count']}")
    console.print(f"Document rows: {result['document_rows']}")
    analysis_path = analysis_dir(config, resolved_chunk_size) / "scope_effect"
    if model is not None:
        analysis_path = analysis_path / model.value
    console.print(f"Outputs: {analysis_path}")


def evidence_structure_command(
    config_path: Path,
    *,
    model: EmbeddingModel | None,
    chunk_size: int | None,
    overwrite: bool,
) -> None:
    config = load_config(config_path)
    ensure_output_directories(config)
    resolved_chunk_size = resolve_chunk_size(config, chunk_size)
    selected_models = None if model is None else (model,)
    result = analyse_evidence_structure(
        config,
        chunk_size=resolved_chunk_size,
        models=selected_models,
        overwrite=overwrite,
    )
    console.print("[green]Evidence-structure analysis complete.[/green]")
    console.print(f"Queries classified: {result['query_count']}")
    console.print(
        f"Confirmatory queries: {result['confirmatory_query_count']}"
    )
    counts = result["confirmatory_structure_counts"]
    for structure, count in counts.items():
        console.print(f"{structure}: {count}")
    output_path = analysis_dir(config, resolved_chunk_size) / "evidence_structure"
    if model is not None:
        output_path = output_path / model.value
    console.print(f"Outputs: {output_path}")



def qasper_challenge_command(config_path: Path, *, overwrite: bool) -> None:
    config = load_config(config_path)
    ensure_output_directories(config)
    result = freeze_cross_section_challenge(config, overwrite=overwrite)
    console.print("[green]Strict cross-section challenge frozen.[/green]")
    counts = result["counts"]
    console.print(f"All strict test questions: {counts['all_strict_test']}")
    console.print(
        f"Cross-model complete: {counts['cross_model_complete']}"
    )
    console.print(f"Cross-model unseen: {counts['cross_model_unseen']}")
    console.print(f"Granite extension: {counts['granite_extension']}")
    console.print(
        f"Outputs: {config.paths.analysis_dir / 'qasper_cross_section_challenge'}"
    )

def qasper_audit_command(config_path: Path, *, overwrite: bool) -> None:
    config = load_config(config_path)
    ensure_output_directories(config)
    result = audit_qasper_collection(config, overwrite=overwrite)
    console.print("[green]Full QASPER evidence audit complete.[/green]")
    console.print(f"Prepared papers: {result['total_documents']}")
    console.print(f"Usable questions: {result['classified_query_count']}")
    console.print(
        "Strict cross-section questions: "
        f"{result['strict_cross_section_query_count']}"
    )
    console.print(
        "Eligible new cross-model strict candidates: "
        f"{result['eligible_new_cross_model_strict_candidate_count']}"
    )
    console.print(
        f"Outputs: {config.paths.analysis_dir / 'qasper_collection_audit'}"
    )


def chunk_size_sensitivity_command(
    config_path: Path,
    *,
    model: EmbeddingModel,
    chunk_sizes: tuple[int, ...],
    overwrite: bool,
) -> None:
    config = load_config(config_path)
    ensure_output_directories(config)
    result = analyse_chunk_size_sensitivity(
        config,
        model=model,
        chunk_sizes=chunk_sizes,
        overwrite=overwrite,
    )
    console.print("[green]Chunk-size sensitivity analysis complete.[/green]")
    console.print(f"Model: {model.value}")
    console.print(f"Chunk sizes: {result['chunk_sizes']}")
    console.print(f"Queries: {result['query_count']}")
    console.print(
        f"Outputs: {config.paths.analysis_dir / 'chunk_size_sensitivity' / model.value}"
    )


def select_chunk_size_command(config_path: Path, *, overwrite: bool) -> None:
    config = load_config(config_path)
    ensure_output_directories(config)
    result = analyse_chunk_size_pilot(config, overwrite=overwrite)
    console.print("[green]Validation-only chunk-size pilot complete.[/green]")
    console.print(f"Selected chunk size: {result['selected_chunk_size_tokens']} tokens")
    console.print(f"Outputs: {config.paths.analysis_dir / 'chunk_size_pilot'}")
