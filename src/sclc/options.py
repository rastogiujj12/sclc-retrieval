from __future__ import annotations

from enum import Enum


class RetrievalCondition(str, Enum):
    """The five retrieval conditions used by the experiment."""

    BM25 = "bm25"
    FIXED_DENSE = "fixed_dense"
    SECTION_ISOLATED = "section_isolated"
    SECTION_CONSTRAINED = "section_constrained"
    GLOBAL = "global"

    @property
    def is_dense(self) -> bool:
        return self is not RetrievalCondition.BM25


class EmbeddingModel(str, Enum):
    """Embedding models configured for the experiment."""

    GRANITE = "granite"
    JINA = "jina"
