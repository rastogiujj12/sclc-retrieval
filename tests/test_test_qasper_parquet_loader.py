from sclc.config import AppConfig


def test_qasper_parquet_urls_are_configured() -> None:
    config = AppConfig.model_validate(
        {
            "project": {"seed": 42},
            "paths": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "profile_dir": "data/profiles",
                "subset_dir": "data/subsets",
                "hf_cache_dir": "cache/huggingface",
            },
            "dataset": {
                "repo_id": "allenai/qasper",
                "subset": "qasper",
                "splits": ["train", "validation", "test"],
                "parquet_files": {
                    "train": "https://huggingface.co/datasets/allenai/qasper/resolve/refs%2Fconvert%2Fparquet/qasper/train/0000.parquet",
                    "validation": "https://huggingface.co/datasets/allenai/qasper/resolve/refs%2Fconvert%2Fparquet/qasper/validation/0000.parquet",
                    "test": "https://huggingface.co/datasets/allenai/qasper/resolve/refs%2Fconvert%2Fparquet/qasper/test/0000.parquet",
                },
            },
            "document": {},
            "chunking": {
                "canonical_tokenizer": "granite",
                "chunk_size_tokens": 512,
                "overlap_tokens": 0,
            },
            "models": {
                "granite": {
                    "model_id": "granite",
                    "max_document_tokens": 32768,
                },
                "jina": {
                    "model_id": "jina",
                    "max_document_tokens": 8192,
                },
            },
            "sampling": {
                "core_documents": 150,
                "granite_extended_documents": 50,
                "minimum_usable_questions": 1,
                "length_bins": 3,
            },
        }
    )

    assert set(config.dataset.parquet_files) == {
        "train",
        "validation",
        "test",
    }
