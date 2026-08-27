from __future__ import annotations

from enum import StrEnum


class RetrievalCondition(StrEnum):
    """The five retrieval conditions used by the experiment."""

    BM25 = "bm25"
    FIXED_DENSE = "fixed_dense"
    SECTION_ISOLATED = "section_isolated"
    SECTION_CONSTRAINED = "section_constrained"
    GLOBAL = "global"

    @property
    def is_dense(self) -> bool:
        return self is not RetrievalCondition.BM25


class EmbeddingModel(StrEnum):
    """Embedding models configured for the experiment."""

    GRANITE = "granite"
    JINA = "jina"
