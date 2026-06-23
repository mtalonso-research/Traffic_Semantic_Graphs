import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import torch


REPO = Path(r"F:\NYCU\nuplan_clean\Traffic_Semantic_Graphs")
EXPERIMENTS = Path(r"F:\NYCU\city_experiments")
CURRENT_BEST = REPO / "models/City/UST/singapore_to_boston/anchor30_tuned_cw12_rescons/singapore_boston_anchor30_tuned_cw12_rescons_seed42_best_model.pt"
SOURCE_AE = REPO / "models/City/BaselineB/singapore_to_boston/singapore_seed42_ae_best_model.pt"
TARGET_AE = REPO / "models/City/BaselineB/singapore_to_boston/boston_seed42_ae_best_model.pt"
CITY_VIEW = REPO / "data/NuPlan/city_views/singapore_to_boston"

ANCHORS = (3, 5, 10, 20, 50)
ALIGNMENTS = (0.001, 0.005, 0.01, 0.02)
CONSISTENCIES = (0.0, 0.001, 0.005)


def slug(value: float) -> str:
    text = f"{value:g}"
    if text == "0":
        return "0"
    return text.replace("0.", "").replace(".", "p")


def parse_output(text: str):
    metrics = {}
    for city in ("singapore", "boston", "overall"):
        for name in ("accuracy", "balanced_accuracy", "macro_f1", "qwk", "ordinal_mae_bins", "loss"):
            match = re.search(rf"eval/{city}/{name}:\s*(-?\d+(?:\.\d+)?)", text)
            if match:
                metrics[f"{city}_{name}"] = float(match.group(1))

    matrices = {}
    for city in ("SINGAPORE", "BOSTON"):
        marker = f"{city} confusion matrix"
        start = text.find(marker)
        rows = []
        if start >= 0:
            for line in text[start:].splitlines()[1:12]:
                match = re.match(r"\s*[0-3]\s*\|\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", line)
                if match:
                    rows.append([int(value) for value in match.groups()])
                if len(rows) == 4:
                    break
        matrices[city.lower()] = rows
        metrics[f"{city.lower()}_prediction_distribution"] = (
            [sum(row[col] for row in rows) for col in range(4)] if len(rows) == 4 else []
        )
    shares = []
    for city in ("singapore", "boston"):
        distribution = metrics.get(f"{city}_prediction_distribution", [])
        if distribution and sum(distribution):
            shares.append(max(distribution) / sum(distribution))
    metrics["max_prediction_share"] = max(shares, default=1.0)
    metrics["class_collapse"] = metrics["max_prediction_share"] >= 0.80
    return metrics, matrices


def base_command(checkpoint: Path):
    return [
        sys.executable,
        "-B",
        "-m",
        "scripts.city_train",
        "ust",
        "--source_city",
        "singapore",
        "--target_city",
        "boston",
        "--city_view_root",
        str(CITY_VIEW),
        "--anchor_strategy",
        "stratified",
        "--load_best_ae_clean",
        "--ae_clean_ckpt_path",
        str(SOURCE_AE),
        "--load_best_ae_noisy",
        "--ae_noisy_ckpt_path",
        str(TARGET_AE),
        "--seed",
        "42",
        "--embed_dim",
        "64",
        "--batch_size",
        "64",
        "--num_workers",
        "0",
        "--patience",
        "15",
        "--best_model_path",
        str(checkpoint),
    ]


def run_process(command, log_path: Path):
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["WANDB_MODE"] = "disabled"
    env["MPLCONFIGDIR"] = str(EXPERIMENTS / ".mplconfig")
    with log_path.open("x", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        captured = []
        for line in process.stdout:
            log.write(line)
            log.flush()
            captured.append(line)
            if "Stage2 Epoch" in line or "evaluation results" in line or "early-stop" in line:
                print(line.rstrip(), flush=True)
        code = process.wait()
    text = "".join(captured)
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}; see {log_path}")
    return text


def evaluate_reference():
    output_dir = EXPERIMENTS / "current_best_reference"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    if result_path.exists():
        print(f"Skipping existing reference: {result_path}")
        return
    log_path = output_dir / "evaluation.log"
    command = base_command(CURRENT_BEST) + ["--anchor_pct", "30", "--evaluate"]
    text = run_process(command, log_path)
    metrics, matrices = parse_output(text)
    payload = {
        "kind": "current_best_reference",
        "anchor_pct": 30,
        "align_weight": 0.5,
        "consistency_weight": 0.5,
        "learning_rate": 0.00005,
        "metrics": metrics,
        "confusion_matrices": matrices,
        "checkpoint": str(CURRENT_BEST),
        "command": subprocess.list2cmdline(command),
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_experiment(anchor: int, align: float, consistency: float):
    name = f"anchor{anchor}_align{slug(align)}_cons{slug(consistency)}"
    output_dir = EXPERIMENTS / name
    result_path = output_dir / "result.json"
    if result_path.exists():
        print(f"Skipping completed run: {name}")
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty experiment folder: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint.pt"
    log_path = output_dir / "run.log"
    command = base_command(checkpoint) + [
        "--anchor_pct",
        str(anchor),
        "--proj_residual",
        "--proj_dropout",
        "0.05",
        "--align_loss_kind",
        "smoothl1",
        "--align_weight",
        str(align),
        "--consistency_kind",
        "kl",
        "--consistency_weight",
        str(consistency),
        "--load_risk_head",
        "--risk_head_ckpt_path",
        str(CURRENT_BEST),
        "--load_proj_noisy",
        "--proj_noisy_ckpt_path",
        str(CURRENT_BEST),
        "--train_noisy_proj_only",
        "--train_stage2",
        "--stage2_epochs",
        "1",
        "--stage2_lr",
        "0.00005",
        "--evaluate",
    ]
    print(f"\n=== {name} ===", flush=True)
    text = run_process(command, log_path)
    metrics, matrices = parse_output(text)
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload = {
        "kind": "sweep",
        "anchor_pct": anchor,
        "align_weight": align,
        "consistency_weight": consistency,
        "learning_rate": 0.00005,
        "metrics": metrics,
        "confusion_matrices": matrices,
        "validation_loss": float(checkpoint_data.get("best_val_risk", float("nan"))),
        "checkpoint": str(checkpoint),
        "command": subprocess.list2cmdline(command),
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if metrics.get("class_collapse"):
        print(f"Class collapse flagged for {name}: {metrics.get('max_prediction_share'):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("reference", "sweep", "all"), default="all")
    parser.add_argument("--max_runs", type=int, default=None)
    args = parser.parse_args()
    for required in (REPO, CURRENT_BEST, SOURCE_AE, TARGET_AE, CITY_VIEW):
        if not required.exists():
            raise FileNotFoundError(required)
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    if args.mode in ("reference", "all"):
        evaluate_reference()
    if args.mode in ("sweep", "all"):
        grid = itertools.product(ANCHORS, ALIGNMENTS, CONSISTENCIES)
        for index, (anchor, align, consistency) in enumerate(grid):
            if args.max_runs is not None and index >= args.max_runs:
                break
            run_experiment(anchor, align, consistency)


if __name__ == "__main__":
    main()
