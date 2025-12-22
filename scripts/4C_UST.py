#!/usr/bin/env python3
"""
End-to-end training:
  (L2D graph -> Encoder_L2D) and (NuPlan graph -> Encoder_NuPlan)
      -> shared ProjectionHead
      -> RiskPredictionHead (supervised only on NuPlan)

Loss:
  total_loss = risk_weight * risk_loss(NuPlan) + kl_weight * KL(proj(L2D), proj(NuPlan))

No reconstruction losses.
"""

import argparse
import os
import sys
import json
import yaml
import random
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_encoding.data_loaders import get_graph_dataset
from src.graph_encoding.autoencoder import (
    HeteroGraphAutoencoder,
    QuantileFeatureQuantizer,
    ProjectionHead,
    batched_graph_embeddings,
)
from src.graph_encoding.risk_prediction import RiskPredictionHead
from src.experiment_utils.C_utils import (set_seed,seed_worker,make_infinite,
                                          _require_dir,_require_file,_print_access,
                                          resolve_paths,diag_gaussian_kl,kl_from_projected)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_task(args):
    set_seed(args.seed)

    # wandb init (only when training)
    wandb_run = None
    if args.use_wandb and args.wandb_mode != "disabled" and args.train:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name or None,
            group=args.wandb_group or None,
            mode=args.wandb_mode,
            config=vars(args),
        )
        # allow config override
        for k, v in wandb.config.items():
            setattr(args, k, v)

    # Model path naming
    side_info_str = "_with_side_info" if args.side_info_path else ""
    pred_tag = "_class" if args.prediction_mode == "classification" else "_reg"
    if os.path.basename(args.best_model_path) == "best_model.pt":
        args.best_model_path = f"./models/end2end_risk/4E{side_info_str}{pred_tag}_best_model.pt"

    # Resolve new dataset structure
    paths = resolve_paths(args.data_root)

    # Print all expected data paths
    for label, key in [
        ("L2D TRAIN graphs", "l2d_train_graphs"),
        ("L2D TRAIN risk", "l2d_train_risk"),
        ("NuPlan TRAIN graphs", "nuplan_train_graphs"),
        ("NuPlan TRAIN risk", "nuplan_train_risk"),
        ("L2D EVAL graphs", "l2d_eval_graphs"),
        ("L2D EVAL risk", "l2d_eval_risk"),
        ("NuPlan EVAL graphs", "nuplan_eval_graphs"),
        ("NuPlan EVAL risk", "nuplan_eval_risk"),
    ]:
        _print_access(label, paths[key])

    _require_dir(paths["l2d_train_graphs"], "L2D training graphs")
    _require_file(paths["l2d_train_risk"], "L2D training risk JSON")
    _require_dir(paths["nuplan_train_graphs"], "NuPlan training graphs")
    _require_file(paths["nuplan_train_risk"], "NuPlan training risk JSON")

    if args.evaluate:
        _require_dir(paths["l2d_eval_graphs"], "L2D evaluation graphs")
        _require_file(paths["l2d_eval_risk"], "L2D evaluation risk JSON")
        _require_dir(paths["nuplan_eval_graphs"], "NuPlan evaluation graphs")
        _require_file(paths["nuplan_eval_risk"], "NuPlan evaluation risk JSON")

    thresholds: Optional[Tuple[float, float, float]] = None
    if args.prediction_mode == "classification":
        if args.class_thresholds is not None:
            parts = [p.strip() for p in args.class_thresholds.split(",") if p.strip() != ""]
            if len(parts) != 3:
                raise ValueError("--class_thresholds must have exactly 3 comma-separated values.")
            thresholds = (float(parts[0]), float(parts[1]), float(parts[2]))

        def risk_to_class(y_float: torch.Tensor) -> torch.Tensor:
            assert thresholds is not None
            t1, t2, t3 = thresholds
            r = y_float.view(-1).float()
            out = torch.zeros_like(r, dtype=torch.long)
            out[r > t1] = 1
            out[r > t2] = 2
            out[r > t3] = 3
            return out

    # ----------------- Load TRAIN datasets -----------------
    print("\nLoading TRAIN datasets...")
    l2d_train_dataset_full = get_graph_dataset(
        root_dir=paths["l2d_train_graphs"],
        mode=args.mode,
        side_information_path=args.side_info_path,
        node_features_to_exclude=args.node_features_to_exclude,
        risk_scores_path=paths["l2d_train_risk"],
    )
    nuplan_train_dataset_full = get_graph_dataset(
        root_dir=paths["nuplan_train_graphs"],
        mode=args.mode,
        side_information_path=None,
        node_features_to_exclude=args.node_features_to_exclude,
        risk_scores_path=paths["nuplan_train_risk"],
    )

    # Quantizers
    print("Fitting quantizers on TRAIN datasets...")
    l2d_quantizer = QuantileFeatureQuantizer(bins=args.quant_bins, node_types=l2d_train_dataset_full.get_metadata()[0])
    l2d_quantizer.fit(l2d_train_dataset_full)

    nuplan_quantizer = QuantileFeatureQuantizer(bins=args.quant_bins, node_types=nuplan_train_dataset_full.get_metadata()[0])
    nuplan_quantizer.fit(nuplan_train_dataset_full)

    # Deterministic train/val splits
    split_gen = torch.Generator().manual_seed(args.seed)

    l2d_val_size = int(args.val_fraction * len(l2d_train_dataset_full))
    l2d_train_size = len(l2d_train_dataset_full) - l2d_val_size
    l2d_train_dataset, l2d_val_dataset = random_split(
        l2d_train_dataset_full, [l2d_train_size, l2d_val_size], generator=split_gen
    )

    nuplan_val_size = int(args.val_fraction * len(nuplan_train_dataset_full))
    nuplan_train_size = len(nuplan_train_dataset_full) - nuplan_val_size
    nuplan_train_dataset, nuplan_val_dataset = random_split(
        nuplan_train_dataset_full, [nuplan_train_size, nuplan_val_size], generator=split_gen
    )

    # Loaders: drop_last=True for KL pairing safety (train + in-training val)
    loader_gen = torch.Generator().manual_seed(args.seed)

    l2d_train_loader = DataLoader(
        l2d_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=loader_gen,
    )
    nuplan_train_loader = DataLoader(
        nuplan_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=loader_gen,
    )

    l2d_val_loader = DataLoader(
        l2d_val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=loader_gen,
    )
    nuplan_val_loader = DataLoader(
        nuplan_val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=loader_gen,
    )

    # ----------------- Models -----------------
    print("\nInitializing models...")
    l2d_side_info_dim = getattr(l2d_train_dataset_full, "side_info_dim", 0) if args.side_info_path else 0
    nuplan_side_info_dim = 0

    l2d_encoder = HeteroGraphAutoencoder(
        metadata=l2d_train_dataset_full.get_metadata(),
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        quantizer_spec=l2d_quantizer.spec(),
        feat_emb_dim=16,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        activation=args.activation,
        dropout_rate=args.dropout_rate,
        side_info_dim=l2d_side_info_dim,
    ).to(device)

    nuplan_encoder = HeteroGraphAutoencoder(
        metadata=nuplan_train_dataset_full.get_metadata(),
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        quantizer_spec=nuplan_quantizer.spec(),
        feat_emb_dim=16,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        activation=args.activation,
        dropout_rate=args.dropout_rate,
        side_info_dim=nuplan_side_info_dim,
    ).to(device)

    # Infer graph embedding dim
    nuplan_encoder.eval()
    with torch.no_grad():
        b0 = next(iter(nuplan_train_loader))
        b0 = nuplan_quantizer.transform_inplace(b0).to(device)
        z0, _, _ = nuplan_encoder(b0)
        g0 = batched_graph_embeddings(z0, b0, nuplan_train_dataset_full.get_metadata(), embed_dim_per_type=args.embed_dim)
        graph_in_dim = int(g0.shape[-1])

    l2d_encoder.eval()
    with torch.no_grad():
        b1 = next(iter(l2d_train_loader))
        b1 = l2d_quantizer.transform_inplace(b1).to(device)
        z1, _, _ = l2d_encoder(b1)
        g1 = batched_graph_embeddings(z1, b1, l2d_train_dataset_full.get_metadata(), embed_dim_per_type=args.embed_dim)
        l2d_graph_in_dim = int(g1.shape[-1])

    if l2d_graph_in_dim != graph_in_dim:
        raise ValueError(
            f"Graph embedding dim mismatch: NuPlan={graph_in_dim}, L2D={l2d_graph_in_dim}. "
            "KL alignment requires identical dims."
        )

    print(f"[info] inferred graph embedding dim = {graph_in_dim}")

    projection_head = ProjectionHead(in_dim=graph_in_dim, proj_dim=args.proj_dim).to(device)

    # Risk head input dim
    if args.kl_assume_unit_var:
        risk_in_dim = args.proj_dim
    else:
        if args.proj_dim % 2 != 0:
            print("[warn] proj_dim is odd but kl_assume_unit_var=False. Falling back to unit-variance KL.")
            args.kl_assume_unit_var = True
            risk_in_dim = args.proj_dim
        else:
            risk_in_dim = args.proj_dim // 2

    output_dim = 1 if args.prediction_mode == "regression" else args.num_classes
    risk_head = RiskPredictionHead(
        input_dim=risk_in_dim,
        hidden_dim=args.risk_hidden_dim,
        output_dim=output_dim,
        mode=args.prediction_mode,
    ).to(device)

    # Loss functions
    risk_loss_fn = nn.MSELoss() if args.prediction_mode == "regression" else nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        list(l2d_encoder.parameters())
        + list(nuplan_encoder.parameters())
        + list(projection_head.parameters())
        + list(risk_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    def forward_graph_embedding(encoder, quantizer, batch, dataset_metadata):
        batch = quantizer.transform_inplace(batch).to(device)
        z_dict, _, _ = encoder(batch)
        graph_emb = batched_graph_embeddings(z_dict, batch, dataset_metadata, embed_dim_per_type=args.embed_dim)
        proj = projection_head(graph_emb)
        return batch, proj

    # ----------------- Train -----------------
    best_val_metric = float("inf")

    if args.train:
        print("\nTraining end-to-end (NuPlan supervised, KL aligns NuPlan↔L2D)...")

        l2d_inf = make_infinite(l2d_train_loader)
        nuplan_inf = make_infinite(nuplan_train_loader)
        steps_per_epoch = max(len(l2d_train_loader), len(nuplan_train_loader))

        for epoch in range(1, args.num_epochs + 1):
            l2d_encoder.train()
            nuplan_encoder.train()
            projection_head.train()
            risk_head.train()

            total_loss = 0.0
            total_risk = 0.0
            total_kl = 0.0
            correct = 0
            seen = 0

            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch:02d} [train]")
            for step in pbar:
                l2d_batch = next(l2d_inf)
                nuplan_batch = next(nuplan_inf)

                optimizer.zero_grad()

                _, l2d_proj = forward_graph_embedding(
                    l2d_encoder, l2d_quantizer, l2d_batch, l2d_train_dataset_full.get_metadata()
                )
                nuplan_batch, nuplan_proj = forward_graph_embedding(
                    nuplan_encoder, nuplan_quantizer, nuplan_batch, nuplan_train_dataset_full.get_metadata()
                )

                # KL loss
                if args.kl_assume_unit_var:
                    zeros_l = torch.zeros_like(l2d_proj)
                    zeros_n = torch.zeros_like(nuplan_proj)
                    kl_loss = diag_gaussian_kl(l2d_proj, zeros_l, nuplan_proj, zeros_n, reduction="mean")
                else:
                    kl_loss = kl_from_projected(l2d_proj, nuplan_proj, reduction="mean")

                # Risk loss (NuPlan only)
                if args.prediction_mode == "regression":
                    y = nuplan_batch.y.view(-1, 1).float().to(device)
                else:
                    if nuplan_batch.y.dtype.is_floating_point:
                        if thresholds is None:
                            raise RuntimeError(
                                "NuPlan batch.y is float but prediction_mode=classification and no --class_thresholds."
                            )
                        y = risk_to_class(nuplan_batch.y.to(device))
                    else:
                        y = nuplan_batch.y.view(-1).long().to(device)

                nuplan_for_risk = nuplan_proj if args.kl_assume_unit_var else nuplan_proj[:, : nuplan_proj.shape[-1] // 2]
                pred = risk_head(nuplan_for_risk)

                risk_loss = risk_loss_fn(pred, y)
                loss = args.risk_weight * risk_loss + args.kl_weight * kl_loss

                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                total_risk += float(risk_loss.item())
                total_kl += float(kl_loss.item())

                if args.prediction_mode == "classification":
                    pred_cls = torch.argmax(pred, dim=-1)
                    correct += int((pred_cls == y).sum().item())
                    seen += int(y.numel())

                pbar.set_postfix(
                    loss=f"{total_loss/(step+1):.4f}",
                    risk=f"{total_risk/(step+1):.4f}",
                    kl=f"{total_kl/(step+1):.4f}",
                    acc=f"{(correct/max(seen,1)):.3f}" if args.prediction_mode == "classification" else "n/a",
                )

            train_acc = (correct / max(seen, 1)) if args.prediction_mode == "classification" else None

            # ----------------- In-training Validation -----------------
            l2d_encoder.eval()
            nuplan_encoder.eval()
            projection_head.eval()
            risk_head.eval()

            def eval_risk(loader, dataset_metadata, quantizer, encoder, name: str) -> Tuple[float, Optional[float]]:
                total = 0.0
                n = 0
                correct_v = 0
                seen_v = 0
                with torch.no_grad():
                    for batch in tqdm(loader, desc=f"Epoch {epoch:02d} [val:{name}]"):
                        batch, proj = forward_graph_embedding(encoder, quantizer, batch, dataset_metadata)

                        if args.prediction_mode == "regression":
                            yv = batch.y.view(-1, 1).float().to(device)
                        else:
                            if batch.y.dtype.is_floating_point:
                                if thresholds is None:
                                    raise RuntimeError(
                                        f"{name} val batch.y is float but prediction_mode=classification and no --class_thresholds."
                                    )
                                yv = risk_to_class(batch.y.to(device))
                            else:
                                yv = batch.y.view(-1).long().to(device)

                        proj_for_risk = proj if args.kl_assume_unit_var else proj[:, : proj.shape[-1] // 2]
                        pv = risk_head(proj_for_risk)
                        lv = risk_loss_fn(pv, yv)

                        total += float(lv.item())
                        n += 1

                        if args.prediction_mode == "classification":
                            pred_cls = torch.argmax(pv, dim=-1)
                            correct_v += int((pred_cls == yv).sum().item())
                            seen_v += int(yv.numel())

                avg = total / max(n, 1)
                acc = (correct_v / max(seen_v, 1)) if args.prediction_mode == "classification" else None
                return avg, acc

            nuplan_val_risk, nuplan_val_acc = eval_risk(
                nuplan_val_loader,
                nuplan_train_dataset_full.get_metadata(),
                nuplan_quantizer,
                nuplan_encoder,
                "NuPlan",
            )

            # L2D val risk is just reported (not used for model selection)
            l2d_val_risk, l2d_val_acc = eval_risk(
                l2d_val_loader,
                l2d_train_dataset_full.get_metadata(),
                l2d_quantizer,
                l2d_encoder,
                "L2D",
            )

            # KL validation (safe due to drop_last=True)
            total_val_kl = 0.0
            val_steps = min(len(l2d_val_loader), len(nuplan_val_loader))
            with torch.no_grad():
                for (l2d_b, nuplan_b) in tqdm(
                    zip(l2d_val_loader, nuplan_val_loader),
                    total=val_steps,
                    desc=f"Epoch {epoch:02d} [val:KL]",
                ):
                    _, l2d_proj_v = forward_graph_embedding(
                        l2d_encoder, l2d_quantizer, l2d_b, l2d_train_dataset_full.get_metadata()
                    )
                    _, nuplan_proj_v = forward_graph_embedding(
                        nuplan_encoder, nuplan_quantizer, nuplan_b, nuplan_train_dataset_full.get_metadata()
                    )

                    if args.kl_assume_unit_var:
                        zeros_l = torch.zeros_like(l2d_proj_v)
                        zeros_n = torch.zeros_like(nuplan_proj_v)
                        klv = diag_gaussian_kl(l2d_proj_v, zeros_l, nuplan_proj_v, zeros_n, reduction="mean")
                    else:
                        klv = kl_from_projected(l2d_proj_v, nuplan_proj_v, reduction="mean")

                    total_val_kl += float(klv.item())

            val_kl = total_val_kl / max(val_steps, 1)

            # Model selection metric (NuPlan supervised + KL)
            val_metric = args.risk_weight * nuplan_val_risk + args.kl_weight * val_kl

            msg = (
                f"Epoch {epoch:02d} | "
                f"NuPlan_val_risk={nuplan_val_risk:.4f} | "
                f"L2D_val_risk={l2d_val_risk:.4f} | "
                f"val_kl={val_kl:.4f} | "
                f"val_metric={val_metric:.4f}"
            )
            if args.prediction_mode == "classification":
                msg += (
                    f" | NuPlan_val_acc={nuplan_val_acc:.4f}"
                    f" | L2D_val_acc={l2d_val_acc:.4f}"
                    f" | train_acc={train_acc:.4f}"
                )
            print(msg)

            if wandb_run is not None:
                payload = {
                    "epoch": epoch,
                    "nuplan_val_risk": nuplan_val_risk,
                    "l2d_val_risk": l2d_val_risk,
                    "val_kl": val_kl,
                    "val_metric": val_metric,
                }
                if args.prediction_mode == "classification":
                    payload.update(
                        {
                            "nuplan_val_acc": nuplan_val_acc,
                            "l2d_val_acc": l2d_val_acc,
                            "train_acc": train_acc,
                        }
                    )
                wandb.log(payload)

            if val_metric < best_val_metric:
                best_val_metric = val_metric
                os.makedirs(os.path.dirname(args.best_model_path), exist_ok=True)
                torch.save(
                    {
                        "l2d_encoder": l2d_encoder.state_dict(),
                        "nuplan_encoder": nuplan_encoder.state_dict(),
                        "projection_head": projection_head.state_dict(),
                        "risk_head": risk_head.state_dict(),
                        "args": vars(args),
                        "graph_in_dim": graph_in_dim,
                    },
                    args.best_model_path,
                )
                print(f"  -> New best model saved to {args.best_model_path}")

    # ----------------- Evaluate -----------------
    if args.evaluate:
        print(f"\nLoading checkpoint: {args.best_model_path}")
        state = torch.load(args.best_model_path, map_location=device, weights_only=False)
        l2d_encoder.load_state_dict(state["l2d_encoder"])
        nuplan_encoder.load_state_dict(state["nuplan_encoder"])
        projection_head.load_state_dict(state["projection_head"])
        risk_head.load_state_dict(state["risk_head"])

        l2d_encoder.eval()
        nuplan_encoder.eval()
        projection_head.eval()
        risk_head.eval()

        print("\nLoading EVAL datasets...")
        l2d_eval_dataset = get_graph_dataset(
            root_dir=paths["l2d_eval_graphs"],
            mode=args.mode,
            side_information_path=args.side_info_path,
            node_features_to_exclude=args.node_features_to_exclude,
            risk_scores_path=paths["l2d_eval_risk"],
        )
        nuplan_eval_dataset = get_graph_dataset(
            root_dir=paths["nuplan_eval_graphs"],
            mode=args.mode,
            side_information_path=None,
            node_features_to_exclude=args.node_features_to_exclude,
            risk_scores_path=paths["nuplan_eval_risk"],
        )

        # EVAL loaders: drop_last=False to evaluate all samples
        l2d_eval_loader = DataLoader(
            l2d_eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            drop_last=False,
            worker_init_fn=seed_worker if args.num_workers > 0 else None,
        )
        nuplan_eval_loader = DataLoader(
            nuplan_eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            drop_last=False,
            worker_init_fn=seed_worker if args.num_workers > 0 else None,
        )

        def eval_split(loader, dataset_metadata, quantizer, encoder, name: str) -> Tuple[float, Optional[float]]:
            total = 0.0
            n = 0
            correct_e = 0
            seen_e = 0
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"[eval:{name}]"):
                    batch, proj = forward_graph_embedding(encoder, quantizer, batch, dataset_metadata)

                    if args.prediction_mode == "regression":
                        yv = batch.y.view(-1, 1).float().to(device)
                    else:
                        if batch.y.dtype.is_floating_point:
                            if thresholds is None:
                                raise RuntimeError(
                                    f"{name} eval batch.y is float but prediction_mode=classification and no --class_thresholds."
                                )
                            yv = risk_to_class(batch.y.to(device))
                        else:
                            yv = batch.y.view(-1).long().to(device)

                    proj_for_risk = proj if args.kl_assume_unit_var else proj[:, : proj.shape[-1] // 2]
                    pv = risk_head(proj_for_risk)
                    lv = risk_loss_fn(pv, yv)

                    total += float(lv.item())
                    n += 1

                    if args.prediction_mode == "classification":
                        pred_cls = torch.argmax(pv, dim=-1)
                        correct_e += int((pred_cls == yv).sum().item())
                        seen_e += int(yv.numel())

            avg = total / max(n, 1)
            acc = (correct_e / max(seen_e, 1)) if args.prediction_mode == "classification" else None
            if acc is None:
                print(f"{name} avg loss: {avg:.4f}")
            else:
                print(f"{name} avg loss: {avg:.4f} | acc: {acc:.4f}")
            return avg, acc

        nuplan_loss, nuplan_acc = eval_split(
            nuplan_eval_loader,
            nuplan_eval_dataset.get_metadata(),
            nuplan_quantizer,
            nuplan_encoder,
            "NuPlan (evaluation_data)",
        )
        l2d_loss, l2d_acc = eval_split(
            l2d_eval_loader,
            l2d_eval_dataset.get_metadata(),
            l2d_quantizer,
            l2d_encoder,
            "L2D (evaluation_data)",
        )

        if args.save_annotations:
            results_path = args.eval_results_path
            results = {}
            if os.path.exists(results_path):
                with open(results_path, "r") as f:
                    results = json.load(f)

            key = "4E_end2end"
            results.setdefault(key, {})
            side = "with_side_info" if args.side_info_path else "without_side_info"

            results[key][f"NuPlan_{side}_loss"] = nuplan_loss
            results[key][f"L2D_{side}_loss"] = l2d_loss
            if args.prediction_mode == "classification":
                results[key][f"NuPlan_{side}_acc"] = nuplan_acc
                results[key][f"L2D_{side}_acc"] = l2d_acc

            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved evaluation results to {results_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end: graph encoders + projection + risk head.")

    # Data root (new structure)
    parser.add_argument("--data_root", type=str, default="data")

    # Data options
    parser.add_argument("--mode", type=str, default="all")
    parser.add_argument("--side_info_path", type=str, default=None)
    parser.add_argument("--node_features_to_exclude", nargs="+", type=str, default=None)

    # Quantization
    parser.add_argument("--quant_bins", type=int, default=32)

    # Model
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--proj_dim", type=int, default=32)
    parser.add_argument("--num_encoder_layers", type=int, default=1)
    parser.add_argument("--num_decoder_layers", type=int, default=1)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--dropout_rate", type=float, default=0.1)

    # Risk head
    parser.add_argument("--risk_hidden_dim", type=int, default=64)
    parser.add_argument("--prediction_mode", type=str, default="regression", choices=["regression", "classification"])
    parser.add_argument("--num_classes", type=int, default=4)

    # If classification labels are continuous floats, provide thresholds to bin them
    # Example: --class_thresholds "0.0043,0.1008,0.3442"
    parser.add_argument("--class_thresholds", type=str, default=None)

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--risk_weight", type=float, default=1.0)
    parser.add_argument("--kl_weight", type=float, default=0.1)

    # KL behavior
    parser.add_argument("--kl_assume_unit_var", action="store_true", default=True)
    parser.add_argument("--no_kl_assume_unit_var", action="store_true")

    # Task flags
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--save_annotations", action="store_true")
    parser.add_argument("--eval_results_path", type=str, default="evaluation_results.json")

    # Outputs / misc
    parser.add_argument("--best_model_path", type=str, default="./models/end2end_risk/best_model.pt")
    parser.add_argument("--seed", type=int, default=42)

    # wandb
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    # Config
    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args()

    if args.no_kl_assume_unit_var:
        args.kl_assume_unit_var = False

    if args.config is not None:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            if not hasattr(args, k):
                print(f"Warning: key '{k}' in config not found in argparse args; ignoring.")
                continue
            default_val = parser.get_default(k)
            current_val = getattr(args, k)
            if current_val == default_val:
                setattr(args, k, v)

    run_task(args)
