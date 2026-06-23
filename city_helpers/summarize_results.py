import csv
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", r"F:\NYCU\city_experiments\.mplconfig")
import matplotlib.pyplot as plt


ROOT = Path(r"F:\NYCU\city_experiments")
SUMMARY = ROOT / "summary.csv"
BEST = ROOT / "best_result.csv"
FIGURE = ROOT / "anchor_sensitivity.png"


def flatten(payload, path):
    metrics = payload.get("metrics", {})
    return {
        "experiment": path.parent.name,
        "kind": payload.get("kind"),
        "anchor_pct": payload.get("anchor_pct"),
        "align_weight": payload.get("align_weight"),
        "consistency_weight": payload.get("consistency_weight"),
        "learning_rate": payload.get("learning_rate"),
        "singapore_accuracy": metrics.get("singapore_accuracy"),
        "singapore_balanced_accuracy": metrics.get("singapore_balanced_accuracy"),
        "singapore_macro_f1": metrics.get("singapore_macro_f1"),
        "singapore_qwk": metrics.get("singapore_qwk"),
        "singapore_ordinal_mae": metrics.get("singapore_ordinal_mae_bins"),
        "boston_accuracy": metrics.get("boston_accuracy"),
        "boston_balanced_accuracy": metrics.get("boston_balanced_accuracy"),
        "boston_macro_f1": metrics.get("boston_macro_f1"),
        "boston_qwk": metrics.get("boston_qwk"),
        "boston_ordinal_mae": metrics.get("boston_ordinal_mae_bins"),
        "overall_accuracy": metrics.get("overall_accuracy"),
        "validation_loss": payload.get("validation_loss"),
        "singapore_confusion_matrix": json.dumps(payload.get("confusion_matrices", {}).get("singapore", [])),
        "boston_confusion_matrix": json.dumps(payload.get("confusion_matrices", {}).get("boston", [])),
        "singapore_prediction_distribution": json.dumps(metrics.get("singapore_prediction_distribution", [])),
        "boston_prediction_distribution": json.dumps(metrics.get("boston_prediction_distribution", [])),
        "max_prediction_share": metrics.get("max_prediction_share"),
        "class_collapse": metrics.get("class_collapse"),
        "checkpoint": payload.get("checkpoint"),
        "command": payload.get("command"),
    }


def write_csv(path, rows):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = [flatten(json.loads(path.read_text(encoding="utf-8")), path) for path in sorted(ROOT.glob("*/result.json"))]
    if not rows:
        raise SystemExit("No result.json files found")
    write_csv(SUMMARY, rows)
    eligible = [row for row in rows if not row["class_collapse"] and row["overall_accuracy"] is not None]
    eligible.sort(
        key=lambda row: (
            row["overall_accuracy"],
            min(row["singapore_macro_f1"], row["boston_macro_f1"]),
            row["singapore_macro_f1"] + row["boston_macro_f1"],
        ),
        reverse=True,
    )
    write_csv(BEST, eligible[:1])

    sweep_rows = [row for row in rows if row["kind"] == "sweep" and not row["class_collapse"]]
    grouped = {}
    for row in sweep_rows:
        grouped.setdefault(row["anchor_pct"], []).append(row)
    anchors = sorted(grouped)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for city, color in (("singapore", "#2563eb"), ("boston", "#dc2626")):
        accuracy = [sum(row[f"{city}_accuracy"] for row in grouped[a]) / len(grouped[a]) for a in anchors]
        macro_f1 = [sum(row[f"{city}_macro_f1"] for row in grouped[a]) / len(grouped[a]) for a in anchors]
        axes[0].plot(anchors, accuracy, "o-", color=color, label=city.title())
        axes[1].plot(anchors, macro_f1, "o-", color=color, label=city.title())
    axes[0].set_ylabel("Mean accuracy")
    axes[1].set_ylabel("Mean macro-F1")
    for axis in axes:
        axis.set_xlabel("anchor_pct")
        axis.set_xticks(anchors)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Singapore → Boston UST anchor sensitivity")
    if FIGURE.exists():
        raise FileExistsError(f"Refusing to overwrite {FIGURE}")
    fig.savefig(FIGURE, dpi=200)
    plt.close(fig)
    cache = Path(os.environ["MPLCONFIGDIR"])
    if cache.exists():
        shutil.rmtree(cache)
    print(f"Wrote {SUMMARY}")
    print(f"Wrote {BEST}")
    print(f"Wrote {FIGURE}")


if __name__ == "__main__":
    main()
