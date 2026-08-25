from __future__ import annotations

from pathlib import Path

import typer

from sclc.commands import (
    analyse_command,
    chunk_command,
    chunk_size_sensitivity_command,
    compare_command,
    evidence_structure_command,
    encode_command,
    evaluate_command,
    prepare_command,
    profile_command,
    qasper_audit_command,
    qasper_challenge_command,
    retrieve_command,
    sample_command,
    select_chunk_size_command,
)
from sclc.options import EmbeddingModel, RetrievalCondition

app = typer.Typer(
    name="sclc",
    help="Section-constrained late-chunking retrieval experiment.",
    no_args_is_help=True,
)


@app.command()
def prepare(
    config: Path = typer.Option(
        Path("configs/base.yaml"),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the YAML configuration.",
    ),
) -> None:
    """Load and reconstruct the QASPER papers."""
    prepare_command(config)


@app.command()
def profile(
    config: Path = typer.Option(
        Path("configs/base.yaml"),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the YAML configuration.",
    ),
) -> None:
    """Count full-document tokens and assign corpus eligibility groups."""
    profile_command(config)


@app.command()
def sample(
    config: Path = typer.Option(
        Path("configs/base.yaml"),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the YAML configuration.",
    ),
) -> None:
    """Select reproducible document-level samples stratified by length."""
    sample_command(config)


@app.command()
def chunk(
    chunk_size: int | None = typer.Option(
        None,
        "--chunk-size",
        min=1,
        help="Chunk size in canonical tokens; defaults to chunking.chunk_size_tokens.",
    ),
    config: Path = typer.Option(
        Path("configs/base.yaml"),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the YAML configuration.",
    ),
) -> None:
    """Construct canonical continuous and section-bounded retrieval units."""
    chunk_command(config, chunk_size=chunk_size)


@app.command()
def encode(
    condition: RetrievalCondition = typer.Option(
        ...,
        "--condition",
        case_sensitive=False,
        help="Representation condition to construct.",
    ),
    model: EmbeddingModel | None = typer.Option(
        None,
        "--model",
        case_sensitive=False,
        help="Required for dense conditions; omit for BM25.",
    ),
    chunk_size: int | None = typer.Option(
        None,
        "--chunk-size",
        min=1,
        help="Chunk size namespace; defaults to chunking.chunk_size_tokens.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Rebuild cached representations that already exist.",
    ),
    config: Path = typer.Option(
        Path("configs/base.yaml"),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the YAML configuration.",
    ),
) -> None:
    """Construct the lexical or dense representation for one condition."""
    encode_command(
        config,
        condition=condition,
        model=model,
        chunk_size=chunk_size,
        overwrite=overwrite,
    )


@app.command()
def retrieve(
    condition: RetrievalCondition = typer.Option(
        ...,
        "--condition",
        case_sensitive=False,
        help="Retrieval condition to rank.",
    ),
    model: EmbeddingModel | None = typer.Option(
        None,
        "--model",
        case_sensitive=False,
        help="Required for dense conditions; omit for BM25.",
    ),
    chunk_size: int | None = typer.Option(
        None,
        "--chunk-size",
        min=1,
        help="Chunk size namespace; defaults to chunking.chunk_size_tokens.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Rebuild cached rankings.",
    ),
    config: Path = typer.Option(
        Path("configs/base.yaml"),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the YAML configuration.",
    ),
) -> None:
    """Rank candidates within each query's associated paper."""
    retrieve_command(
        config,
        condition=condition,
        model=model,
        chunk_size=chunk_size,
        overwrite=overwrite,
    )


@app.command()
def evaluate(
    condition: RetrievalCondition = typer.Option(
        ...,
        "--condition",
        case_sensitive=False,
        help="Retrieval condition to evaluate.",
    ),
    model: EmbeddingModel | None = typer.Option(
        None,
        "--model",
        case_sensitive=False,
        help="Required for dense conditions; omit for BM25.",
    ),
    chunk_size: int | None = typer.Option(
        None,
        "--chunk-size",
        min=1,
        help="Chunk size namespace; defaults to chunking.chunk_size_tokens.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Rebuild cached metric outputs.",
    ),
    config: Path = typer.Option(
        Path("configs/base.yaml"),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the YAML configuration.",
    ),
) -> None:
    """Evaluate rankings against alternative and union QASPER evidence."""
    evaluate_command(
        config,
        condition=condition,
        model=model,
        chunk_size=chunk_size,
        overwrite=overwrite,
    )


@app.command()
def compare(
    model: EmbeddingModel | None = typer.Option(
        None,
        "--model",
        case_sensitive=False,
        help=(
            "Restrict comparisons to one embedding model. Omit only after all "
            "configured models have been evaluated."
        ),
    ),
    chunk_size: int | None = typer.Option(
        None,
        "--chunk-size",
        min=1,
        help="Selected chunk size namespace.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Rebuild cached bootstrap comparisons.",
    ),
    config: Path = typer.Option(
        Path("configs/base.yaml"),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the YAML configuration.",
    ),
) -> None:
    """Run document-level paired bootstrap comparisons and Holm correction."""
    compare_command(
        config, model=model, chunk_size=chunk_size, overwrite=overwrite
    )


@app.command("analyse")
def analyse(
    model: EmbeddingModel | None = typer.Option(
        None,
        "--model",
        case_sensitive=False,
        help=(
            "Restrict scope-effect analysis to one embedding model. Omit only "
            "after all configured models have been evaluated."
        ),
    ),
    chunk_size: int | None = typer.Option(
        None,
        "--chunk-size",
        min=1,
        help="Selected chunk size namespace.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Rebuild cached scope-effect analysis outputs.",
    ),
    config: Path = typer.Option(
        Path("configs/base.yaml"),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the YAML configuration.",
    ),
) -> None:
    """Relate section-constrained minus global performance to paper structure."""
    analyse_command(
        config, model=model, chunk_size=chunk_size, overwrite=overwrite
    )


@app.command("evidence-structure")
def evidence_structure(
    model: EmbeddingModel | None = typer.Option(
        None,
        "--model",
        case_sensitive=False,
        help=(
            "Restrict evidence-structure analysis to one embedding model. "
            "Omit only after all configured models have been evaluated."
        ),
    ),
    chunk_size: int | None = typer.Option(
        None,
        "--chunk-size",
        min=1,
        help="Selected chunk size namespace.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Rebuild cached evidence-structure outputs.",
    ),
    config: Path = typer.Option(
        Path("configs/base.yaml"),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the YAML configuration.",
    ),
) -> None:
    """Analyse retrieval by single-, same-section-, and cross-section evidence."""
    evidence_structure_command(
        config, model=model, chunk_size=chunk_size, overwrite=overwrite
    )


@app.command("qasper-audit")
def qasper_audit(
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Rebuild the complete QASPER evidence-structure audit.",
    ),
    config: Path = typer.Option(
        Path("configs/base.yaml"),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the YAML configuration.",
    ),
) -> None:
    """Audit cross-section evidence across the complete prepared QASPER corpus."""
    qasper_audit_command(config, overwrite=overwrite)


@app.command("qasper-challenge")
def qasper_challenge(
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Rebuild the frozen strict cross-section challenge files.",
    ),
    config: Path = typer.Option(
        Path("configs/base.yaml"),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the YAML configuration.",
    ),
) -> None:
    """Freeze test-only challenge manifests and a blinded review sheet."""
    qasper_challenge_command(config, overwrite=overwrite)


@app.command("chunk-size-sensitivity")
def chunk_size_sensitivity(
    model: EmbeddingModel = typer.Option(
        ...,
        "--model",
        case_sensitive=False,
        help="Embedding model whose completed evaluations should be analysed.",
    ),
    chunk_sizes: str = typer.Option(
        "128,256,512",
        "--chunk-sizes",
        help="Comma-separated chunk sizes with completed dense evaluations.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Rebuild cached chunk-size sensitivity outputs.",
    ),
    config: Path = typer.Option(
        Path("configs/base.yaml"),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the YAML configuration.",
    ),
) -> None:
    """Compare dense retrieval scope across multiple chunk sizes."""
    try:
        parsed_sizes = tuple(
            int(value.strip())
            for value in chunk_sizes.split(",")
            if value.strip()
        )
    except ValueError as error:
        raise typer.BadParameter(
            "--chunk-sizes must be a comma-separated list of integers"
        ) from error
    if len(parsed_sizes) < 2:
        raise typer.BadParameter("--chunk-sizes must contain at least two sizes")
    chunk_size_sensitivity_command(
        config,
        model=model,
        chunk_sizes=parsed_sizes,
        overwrite=overwrite,
    )


@app.command("select-chunk-size")
def select_chunk_size(
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Rebuild the validation-only chunk-size pilot analysis.",
    ),
    config: Path = typer.Option(
        Path("configs/base.yaml"),
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the YAML configuration.",
    ),
) -> None:
    """Select one chunk size using validation-only BM25 and Granite dense pilots."""
    select_chunk_size_command(config, overwrite=overwrite)


if __name__ == "__main__":
    app()
