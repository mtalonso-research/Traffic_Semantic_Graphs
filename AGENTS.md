# Repository Guidelines

## Project Structure & Module Organization
Traffic Semantic Graphs keeps bulky datasets under `data/` (git-ignored) with subfolders that mirror the README tree; preserve the same hierarchy so orchestration scripts can resolve `raw/`, `processed/`, and `graphical_final/`. Core logic lives in `src/`: `data_processing/` hosts dataset loaders and transformers, `utils.py` centralizes math helpers, and `visualizations.py` handles plotting. Workflow entry points stay inside `scripts/` with numeric prefixes (`1A_*`, `1B_*`, `1C_*`); new orchestration or visualization scripts belong here, while checkpoints reside in `ml-depth-pro/` or `models/`.

## Build, Test, and Development Commands
- `conda env create -f sem_graphs_environment.yml` (or `nuplan_environment.yml`) reproduces the CUDA/pandas stack; run `conda activate traffic-semantic-graphs` afterward.
- `python -m scripts.1A_l2d_processing --all` executes the entire L2D pipeline; narrow scope with `--min_ep/--max_ep`.
- `python -m scripts.1B_nup_processing --city boston --tags` runs a NuPlan slice; omit step flags for the full pass.
- `python -m scripts.1C_final_processing --process_l2d` materializes the canonical graphs consumed downstream.
- `python -m scripts.scene_visualizer --help` lists options before writing QA frames into `figures/`.

## Coding Style & Naming Conventions
Use Python 3.10, 4-space indentation, and snake_case for files, modules, and functions. Mirror the existing `nup_process_*` naming when adding dataset-specific helpers, prefix private utilities with `_`, and add lightweight type hints for anything returning arrays or frames. Favor vectorized NumPy/Pandas operations, keep plotters pure (no file writes inside `src/visualizations.py`), and name artifacts with lowercase hyphenated slugs such as `l2d-tags-final.csv`.

## Testing Guidelines
Smoke-test every pipeline against a tiny episode range before large runs. Add notebook-based validations to `risk-testing-*.ipynb` or lightweight `pytest` modules under `tests/` (e.g., `tests/test_l2d_process.py`) when logic moves into `src/`. Record the command and dataset slice used, and examine products written to `data/temporary_data/` or `figures/` for regression cues.

## Commit & Pull Request Guidelines
Commits in this repo favor short, imperative summaries ("lane detection for l2d"); match that style and keep each commit scoped to one functional change. Pull requests should describe motivation, list the exact commands executed, call out any data dependencies that cannot ship with the repo, and attach preview images for visualization updates. Link related NuPlan/L2D issues when available and document cleanup steps if a script rewrites artifacts.

## Security & Data Handling
Never commit proprietary datasets or secrets: keep heavy files under the ignored `data/` tree, store tokens in local `.env` files, and scrub GPS/PII columns before sharing notebooks. Reference download locations instead of uploading binaries.
