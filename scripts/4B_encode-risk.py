import argparse
import os
import sys
import json
import yaml
import random
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_encoding.data_loaders import get_graph_dataset
from src.graph_encoding.autoencoder import (
    HeteroGraphAutoencoder,
    batched_graph_embeddings,
    QuantileFeatureQuantizer,
)
from src.graph_encoding.risk_prediction import RiskPredictionHead
from src.experiment_utils.B_utils import (set_seed, seed_worker,risk_to_class_safe,
                                          infer_graph_emb_dim,apply_yaml_overrides,
                                          enforce_dataset_cli_wins,resolve_paths,
                                          _format_confusion_matrix,_print_access,_require_dir,
                                          _require_file)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_task(args: argparse.Namespace) -> None:
    # W&B init:
    # - --wandb: log a normal run
    # - --sweep: allow sweep overrides
    wandb_run = None
    if args.wandb or args.sweep:
        wandb_run = wandb.init(config=vars(args))
        if args.sweep:
            for k, v in wandb.config.items():
                if hasattr(args, k):
                    setattr(args, k, v)

    enforce_dataset_cli_wins(args)
    set_seed(args.seed)
    args.data_root = os.path.abspath(args.data_root)

    if args.l2d == args.nup:
        raise SystemExit("Specify exactly one dataset: --l2d or --nup (not both, not neither).")

    dataset_name = "L2D" if args.l2d else "NuPlan"

    side_info_str = "_with_side_info" if args.with_side_information else ""

    # Add prediction-mode tag to filename
    pred_tag = "_class" if args.prediction_mode == "classification" else "_reg"

    if os.path.basename(args.best_model_path) == "best_model.pt":
        best_model_path = f"./models/risk_predictor/4A_{dataset_name}{side_info_str}{pred_tag}_best_model.pt"
    else:
        best_model_path = args.best_model_path

    paths = resolve_paths(args, dataset_name)

    # ---- Training dataset: graphs + risk scores
    _print_access("TRAIN graphs root", paths["train_graph_root"])
    _print_access("TRAIN risk scores", paths["train_risk_path"])
    _require_dir(paths["train_graph_root"], "Training graphs")
    _require_file(paths["train_risk_path"], "Training risk scores JSON")

    # Side info only for L2D if requested
    side_information_path = None
    if args.with_side_information and args.l2d:
        side_information_path = os.path.join(args.data_root, "training_data", "L2D", "L2D_frame_embs")
        _print_access("Side information", side_information_path)

    print("Loading TRAIN dataset...")
    train_dataset_full = get_graph_dataset(
        root_dir=paths["train_graph_root"],
        mode=args.mode,
        side_information_path=side_information_path,
        risk_scores_path=paths["train_risk_path"],
    )

    print("Fitting quantizer on TRAIN dataset...")
    quantizer = QuantileFeatureQuantizer(bins=args.quant_bins, node_types=train_dataset_full.get_metadata()[0])
    quantizer.fit(train_dataset_full)

    # Deterministic split
    val_size = int(args.val_fraction * len(train_dataset_full))
    train_size = len(train_dataset_full) - val_size
    split_gen = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size], generator=split_gen)

    loader_gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=loader_gen,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=loader_gen,
    )

    # Models
    print("Initializing models...")
    side_info_dim = getattr(train_dataset_full, "side_info_dim", 0) if args.with_side_information else 0
    metadata = train_dataset_full.get_metadata()

    encoder = HeteroGraphAutoencoder(
        metadata=metadata,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        quantizer_spec=quantizer.spec(),
        feat_emb_dim=16,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=1,
        activation=args.activation,
        dropout_rate=args.dropout_rate,
        side_info_dim=side_info_dim,
    ).to(device)

    graph_emb_dim = infer_graph_emb_dim(
        encoder, quantizer, train_loader, metadata, embed_dim_per_type=args.embed_dim
    )
    print(f"[info] inferred graph_emb_dim = {graph_emb_dim}")

    output_dim = 1 if args.prediction_mode == "regression" else args.num_classes
    prediction_head = RiskPredictionHead(
        input_dim=graph_emb_dim,
        hidden_dim=args.risk_hidden_dim,
        output_dim=output_dim,
        mode=args.prediction_mode,
    ).to(device)

    loss_fn = nn.MSELoss() if args.prediction_mode == "regression" else nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(prediction_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    def forward_batch(batch):
        batch = quantizer.transform_inplace(batch).to(device)
        z_dict, _, _ = encoder(batch)
        g = batched_graph_embeddings(z_dict, batch, metadata, embed_dim_per_type=args.embed_dim)
        return batch, g

    # ---------------- TRAIN ----------------
    best_val_loss = float("inf")

    if args.train:
        print(f"Training baseline (encoder→graph_emb→risk_head) on {dataset_name}...")
        for epoch in range(1, args.num_epochs + 1):
            encoder.train()
            prediction_head.train()

            total = 0.0
            correct = 0
            seen = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch:02d} [train]"):
                optimizer.zero_grad()
                batch, g = forward_batch(batch)
                pred = prediction_head(g)

                if args.prediction_mode == "regression":
                    target = batch.y.view(-1, 1).float()
                    loss = loss_fn(pred, target)
                else:
                    target = risk_to_class_safe(batch.y)
                    loss = loss_fn(pred, target)

                    pred_cls = torch.argmax(pred, dim=-1)
                    correct += int((pred_cls == target).sum().item())
                    seen += int(target.numel())

                loss.backward()
                optimizer.step()
                total += float(loss.item())

            train_loss = total / max(len(train_loader), 1)
            train_acc = (correct / max(seen, 1)) if args.prediction_mode == "classification" else None

            # Validation
            encoder.eval()
            prediction_head.eval()
            vtotal = 0.0
            vcorrect = 0
            vseen = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch:02d} [val]"):
                    batch, g = forward_batch(batch)
                    pred = prediction_head(g)

                    if args.prediction_mode == "regression":
                        target = batch.y.view(-1, 1).float()
                        vtotal += float(loss_fn(pred, target).item())
                    else:
                        target = risk_to_class_safe(batch.y)
                        vtotal += float(loss_fn(pred, target).item())

                        pred_cls = torch.argmax(pred, dim=-1)
                        vcorrect += int((pred_cls == target).sum().item())
                        vseen += int(target.numel())

            val_loss = vtotal / max(len(val_loader), 1)
            val_acc = (vcorrect / max(vseen, 1)) if args.prediction_mode == "classification" else None

            if args.prediction_mode == "classification":
                print(
                    f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
                    f"| train_acc={train_acc:.4f} | val_acc={val_acc:.4f}"
                )
            else:
                print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

            if wandb_run is not None:
                payload = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
                if args.prediction_mode == "classification":
                    payload.update({"train_acc": train_acc, "val_acc": val_acc})
                wandb.log(payload)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                torch.save(
                    {
                        "encoder_state_dict": encoder.state_dict(),
                        "prediction_head_state_dict": prediction_head.state_dict(),
                        "best_val_loss": best_val_loss,
                        "graph_emb_dim": graph_emb_dim,
                        "dataset_name": dataset_name,
                        "args": dict(vars(args)),
                    },
                    best_model_path,
                )
                print(f"  -> New best model saved to {best_model_path}")

    # ---------------- EVALUATE ----------------
    if args.evaluate:
        print(f"Evaluating baseline on {dataset_name}...")

        _require_file(best_model_path, "Checkpoint (--best_model_path)")

        try:
            ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(best_model_path, map_location=device)

        encoder.load_state_dict(ckpt["encoder_state_dict"])
        prediction_head.load_state_dict(ckpt["prediction_head_state_dict"])
        encoder.eval()
        prediction_head.eval()

        _print_access("EVAL graphs root", paths["eval_graph_root"])
        _print_access("EVAL risk scores", paths["eval_risk_path"])
        _require_dir(paths["eval_graph_root"], "Evaluation graphs")
        _require_file(paths["eval_risk_path"], "Evaluation risk scores JSON")

        print("Loading EVAL dataset...")
        eval_dataset_full = get_graph_dataset(
            root_dir=paths["eval_graph_root"],
            mode=args.mode,
            side_information_path=side_information_path if (args.with_side_information and args.l2d) else None,
            risk_scores_path=paths["eval_risk_path"],
        )

        eval_loader = DataLoader(
            eval_dataset_full,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker if args.num_workers > 0 else None,
        )

        total = 0.0
        n = 0
        correct = 0
        seen = 0

        # Confusion matrix state (classification only)
        num_classes = int(args.num_classes)
        cm = np.zeros((num_classes, num_classes), dtype=np.int64) if args.prediction_mode == "classification" else None

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"[eval:{dataset_name}]"):
                batch, g = forward_batch(batch)
                pred = prediction_head(g)

                if args.prediction_mode == "regression":
                    target = batch.y.view(-1, 1).float()
                    total += float(loss_fn(pred, target).item())
                else:
                    target = risk_to_class_safe(batch.y)
                    total += float(loss_fn(pred, target).item())

                    pred_cls = torch.argmax(pred, dim=-1)
                    correct += int((pred_cls == target).sum().item())
                    seen += int(target.numel())

                    # update confusion matrix
                    t = target.detach().cpu().numpy().astype(np.int64)
                    p = pred_cls.detach().cpu().numpy().astype(np.int64)
                    for ti, pi in zip(t, p):
                        if 0 <= ti < num_classes and 0 <= pi < num_classes:
                            cm[ti, pi] += 1

                n += 1

        avg_loss = total / max(n, 1)

        if args.prediction_mode == "classification":
            eval_acc = correct / max(seen, 1)
            print(f"{dataset_name} evaluation avg loss: {avg_loss:.4f} | eval_acc: {eval_acc:.4f}")
            print("\nConfusion matrix (rows=true, cols=pred):")
            print(_format_confusion_matrix(cm))

            if wandb_run is not None:
                # Log as a W&B plot if available + raw matrix as a table-friendly artifact
                try:
                    wandb.log(
                        {
                            "eval_loss": avg_loss,
                            "eval_acc": eval_acc,
                            "confusion_matrix": wandb.plot.confusion_matrix(
                                probs=None,
                                y_true=np.repeat(np.arange(num_classes), cm.sum(axis=1)),
                                preds=None,
                                class_names=[str(i) for i in range(num_classes)],
                            ),
                        }
                    )
                except Exception:
                    # Fallback: log raw matrix + metrics
                    wandb.log(
                        {
                            "eval_loss": avg_loss,
                            "eval_acc": eval_acc,
                            "confusion_matrix_raw": cm.tolist(),
                        }
                    )
        else:
            print(f"{dataset_name} evaluation avg loss: {avg_loss:.4f}")
            if wandb_run is not None:
                wandb.log({"eval_loss": avg_loss})

        if args.save_annotations:
            results_path = "evaluation_results.json"
            results = {}
            if os.path.exists(results_path):
                with open(results_path, "r") as f:
                    results = json.load(f)

            results.setdefault("4A", {})
            key = (
                "L2D_with_side_info"
                if (args.l2d and args.with_side_information)
                else ("L2D_without_side_info" if args.l2d else "NuPlan")
            )
            results["4A"][key] = avg_loss
            if args.prediction_mode == "classification":
                results["4A"][key + "_acc"] = eval_acc
                results["4A"][key + "_confusion_matrix"] = cm.tolist()

            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved evaluation results to {results_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline: train encoder + risk head end-to-end on one dataset."
    )

    parser.add_argument("--data_root", type=str, default="data")

    # Dataset select
    parser.add_argument("--l2d", action="store_true")
    parser.add_argument("--nup", action="store_true")
    parser.add_argument("--with_side_information", action="store_true")

    # Model
    parser.add_argument("--mode", type=str, default="all")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--num_encoder_layers", type=int, default=1)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--dropout_rate", type=float, default=0.1)

    # Risk head
    parser.add_argument("--risk_hidden_dim", type=int, default=64)
    parser.add_argument("--prediction_mode", type=str, default="classification", choices=["regression", "classification"])
    parser.add_argument("--num_classes", type=int, default=4)

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--quant_bins", type=int, default=32)

    # Task
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--save_annotations", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")

    # Output / misc
    parser.add_argument("--best_model_path", type=str, default="./models/risk_predictor/best_model.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_config", type=str, default=None)

    args = parser.parse_args()
    args = apply_yaml_overrides(parser, args)
    run_task(args)
