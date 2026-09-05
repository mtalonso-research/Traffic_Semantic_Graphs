# Progress Report — Traffic Semantic Graphs

**Prepared for:** Team sync · June 23, 2026
**Project:** Risk prediction from traffic semantic graphs (NuPlan / L2D)

## TL;DR

This cycle I built out the **cross-city domain-transfer** track (Singapore → Boston) on top of the existing noise-robustness work, and ran a full hyperparameter sweep to try to push the UST model past its current best. The sweep did **not** beat the established anchor-30 reference (47% overall), but it cleanly mapped the anchor-budget / loss-weight tradeoff and confirmed real-time inference (~8 ms/graph). The anchor-30 tuned model remains the one to beat.

## What the project does

We turn each driving scene into a semantic graph and predict an ordinal **risk class (4 levels)**. The research angle is robustness: keeping risk prediction accurate when graphs are noisy or come from a different domain (city) than the model was trained on. The method is **UST** (uncertainty-aware semantic alignment) — align a small number of labeled "anchor" samples from the target domain against the source domain, with a learned projection plus alignment and consistency losses.

## Done this cycle

- Stood up the **city-split pipeline**: `city_split_processing.py`, `unique_cities.py`, `city_train.py`, plus helpers (`sweep_city.py`, `summarize_results.py`, `measure_latency.py`, `repo_guard.py`).
- Ran a **Singapore → Boston anchor sensitivity sweep**: anchor_pct ∈ {5, 10, 20, 50} × align_weight ∈ {0.001, 0.005, 0.01, 0.02} × consistency_weight ∈ {0, 0.001, 0.005} — warm-started from the anchor-30 risk head, retraining only the Boston (noisy) projection.
- **Aggregated results** into `summary.csv` / `best_result.csv` and an anchor-sensitivity figure.
- **Measured latency** of the deployed checkpoint.

## Key results

Cross-city accuracy (Singapore → Boston), 4-class ordinal risk:

| Config | Anchor % | Singapore acc | Boston acc | Overall | Boston QWK |
|---|---|---|---|---|---|
| **Current best (reference)** | 30 | 0.45 | 0.49 | **0.47** | 0.51 |
| Best from this sweep (`anchor20_align0.005_cons0.001`) | 20 | 0.40 | 0.51 | 0.455 | 0.51 |
| Typical anchor-50 | 50 | 0.31 | ~0.42 | ~0.36 | ~0.50 |

- **No swept config beat the anchor-30 reference (0.47 overall).** Best sweep result was 0.455.
- **Anchor budget is not monotonic in this warm-start setup.** 20% is the sweet spot among swept values; 50% degrades (Singapore drops to 31%, Boston predictions collapse toward the majority class).
- **Loss weights had small effect** relative to anchor count — alignment/consistency tuning moved Boston accuracy by only a few points.
- **Latency:** ~8.2 ms median per graph (p95 9.6 ms) on an RTX 3050 laptop GPU → ~120 graphs/sec, comfortably real-time.

## Talking points / open questions

- **Accuracy ceiling (~47%) on a 4-class problem** — better than chance but modest. Worth deciding whether the headline metric should be accuracy, QWK (ordinal), or macro-F1, since the classes are imbalanced and ordinal.
- **Warm-start limited the sweep.** Only the Boston projection was retrained (1 stage-2 epoch), so Singapore accuracy was frozen per anchor level. A full retrain per config might tell a different story but costs more compute.
- **Single seed (42).** Results aren't yet averaged over seeds, so small gaps (e.g. 0.455 vs 0.47) may be within noise.

## Suggested next steps

1. Run a few **full-retrain** configs around the anchor-20–30 sweet spot rather than warm-start-only.
2. **Multi-seed** the top 2–3 configs to confirm the reference really wins.
3. Extend to **other city pairs** (e.g. Boston → Pittsburgh, Las Vegas) to test generality.
4. Try **more stage-2 epochs** (currently 1) and class-imbalance handling to fight majority-class collapse at high anchor %.

---
*Source: `city_experiments/summary.csv`, `best_result.csv`, `latency_report.csv`, and `README.md` in the Traffic_Semantic_Graphs repo.*
