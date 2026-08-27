# Reproducibility snapshot

Run `python scripts/capture_environment.py --config configs/base.yaml` on the experiment machine before the archival release. The generated JSON records the local software/CUDA/GPU environment, Git state, frozen-input hashes, dataset configuration, and configured model metadata. Historical model commit hashes are not invented when they cannot be recovered.
