"""
Unified pipeline script: (Encoder+Projection) + (Risk head from embeddings)

Stages (each controlled by flags):
  A) --train_encoder
     Train two autoencoders (L2D, NuPlan) + a shared projection head using:
        recon_loss(L2D) + recon_loss(NuPlan) + kl_weight * KL(proj(L2D_emb), proj(NuPlan_emb))

  B) --extract_embeddings
     Extract projected graph embeddings (after ProjectionHead) for:
        - TRAIN split: data/training_data/<DATASET>/graphs
        - EVAL  split: data/evaluation_data/<DATASET>/graphs
     and save to args.embedding_dir with predictable filenames.

  C) --train_risk
     Train RiskPredictionHead on NuPlan TRAIN embeddings (supervised),
     with an internal train/val split.

  D) --evaluate_risk
     Evaluate the risk head on:
        - NuPlan EVAL embeddings
        - L2D   EVAL embeddings
     Logs loss (+ accuracy/confusion matrix for classification).

"""

import argparse
import os
import sys
import json
import random
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

import wandb
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_encoding.data_loaders import get_graph_dataset
from src.graph_encoding.autoencoder import (
    HeteroGraphAutoencoder,
    feature_loss,
    edge_loss,
    QuantileFeatureQuantizer,
    ProjectionHead,
    kl_divergence_between_gaussians,
    batched_graph_embeddings,
)
from src.graph_encoding.risk_prediction import RiskPredictionHead
from src.experiment_utils.D_utils import (set_seed,seed_worker,_require_dir,_require_file,
                                          _print_access,episode_ids_from_batch,risk_to_class_safe,
                                          compute_confusion_matrix,apply_yaml_overrides,resolve_paths,
                                          infer_graph_emb_dim)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    # wandb: one run can cover any combination of stages; log when enabled.
    wandb_run = None
    if args.use_wandb and args.wandb_mode != "disabled":
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name or None,
            group=args.wandb_group or None,
            mode=args.wandb_mode,
            config=vars(args),
        )
        # if running sweeps, allow overrides
        for k, v in wandb.config.items():
            if hasattr(args, k):
                setattr(args, k, v)

    # ---- resolve dataset paths
    l2d_paths = resolve_paths(args.data_root, "L2D")
    nup_paths = resolve_paths(args.data_root, "NuPlan")

    _print_access("L2D TRAIN graphs", l2d_paths["train_graph_root"])
    _print_access("L2D TRAIN risk",   l2d_paths["train_risk_path"])
    _print_access("L2D EVAL  graphs", l2d_paths["eval_graph_root"])
    _print_access("L2D EVAL  risk",   l2d_paths["eval_risk_path"])

    _print_access("NuPlan TRAIN graphs", nup_paths["train_graph_root"])
    _print_access("NuPlan TRAIN risk",   nup_paths["train_risk_path"])
    _print_access("NuPlan EVAL  graphs", nup_paths["eval_graph_root"])
    _print_access("NuPlan EVAL  risk",   nup_paths["eval_risk_path"])

    # Side info path (only used for L2D if provided)
    l2d_side_information_path = None
    if args.side_info_path is not None:
        l2d_side_information_path = args.side_info_path
        _print_access("L2D side info", l2d_side_information_path)

    # ---- Build TRAIN datasets (used by encoder training and/or train embeddings)
    _require_dir(l2d_paths["train_graph_root"], "L2D TRAIN graphs")
    _require_file(l2d_paths["train_risk_path"], "L2D TRAIN risk JSON")
    _require_dir(nup_paths["train_graph_root"], "NuPlan TRAIN graphs")
    _require_file(nup_paths["train_risk_path"], "NuPlan TRAIN risk JSON")

    l2d_train_full = get_graph_dataset(
        root_dir=l2d_paths["train_graph_root"],
        mode=args.mode,
        side_information_path=l2d_side_information_path,
        node_features_to_exclude=args.node_features_to_exclude,
        risk_scores_path=l2d_paths["train_risk_path"],
    )
    nup_train_full = get_graph_dataset(
        root_dir=nup_paths["train_graph_root"],
        mode=args.mode,
        side_information_path=None,
        node_features_to_exclude=args.node_features_to_exclude,
        risk_scores_path=nup_paths["train_risk_path"],
    )

    # Quantizers (fit on TRAIN only)
    l2d_quant = QuantileFeatureQuantizer(bins=args.quant_bins, node_types=l2d_train_full.get_metadata()[0])
    nup_quant = QuantileFeatureQuantizer(bins=args.quant_bins, node_types=nup_train_full.get_metadata()[0])
    print("Fitting quantizers on TRAIN datasets...")
    l2d_quant.fit(l2d_train_full)
    nup_quant.fit(nup_train_full)

    # Deterministic train/val split for encoder training and for risk-head-from-train-embeddings
    split_gen = torch.Generator().manual_seed(args.seed)

    l2d_train_size = int((1 - args.val_fraction) * len(l2d_train_full))
    l2d_val_size = len(l2d_train_full) - l2d_train_size
    l2d_train_ds, l2d_val_ds = random_split(l2d_train_full, [l2d_train_size, l2d_val_size], generator=split_gen)

    nup_train_size = int((1 - args.val_fraction) * len(nup_train_full))
    nup_val_size = len(nup_train_full) - nup_train_size
    nup_train_ds, nup_val_ds = random_split(nup_train_full, [nup_train_size, nup_val_size], generator=split_gen)

    loader_gen = torch.Generator().manual_seed(args.seed)
    common_loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=loader_gen,
    )

    # drop_last=True for paired KL batches during encoder training
    l2d_train_loader_drop = DataLoader(l2d_train_ds, shuffle=True, drop_last=True, **common_loader_kwargs)
    nup_train_loader_drop = DataLoader(nup_train_ds, shuffle=True, drop_last=True, **common_loader_kwargs)
    l2d_val_loader_drop = DataLoader(l2d_val_ds, shuffle=False, drop_last=True, **common_loader_kwargs)
    nup_val_loader_drop = DataLoader(nup_val_ds, shuffle=False, drop_last=True, **common_loader_kwargs)

    # non-drop loaders for embedding extraction
    l2d_train_loader_full = DataLoader(l2d_train_full, shuffle=False, drop_last=False, **common_loader_kwargs)
    nup_train_loader_full = DataLoader(nup_train_full, shuffle=False, drop_last=False, **common_loader_kwargs)

    # ---- Build EVAL datasets (for embedding extraction and risk evaluation)
    _require_dir(l2d_paths["eval_graph_root"], "L2D EVAL graphs")
    _require_file(l2d_paths["eval_risk_path"], "L2D EVAL risk JSON")
    _require_dir(nup_paths["eval_graph_root"], "NuPlan EVAL graphs")
    _require_file(nup_paths["eval_risk_path"], "NuPlan EVAL risk JSON")

    l2d_eval_full = get_graph_dataset(
        root_dir=l2d_paths["eval_graph_root"],
        mode=args.mode,
        side_information_path=l2d_side_information_path,
        node_features_to_exclude=args.node_features_to_exclude,
        risk_scores_path=l2d_paths["eval_risk_path"],
    )
    nup_eval_full = get_graph_dataset(
        root_dir=nup_paths["eval_graph_root"],
        mode=args.mode,
        side_information_path=None,
        node_features_to_exclude=args.node_features_to_exclude,
        risk_scores_path=nup_paths["eval_risk_path"],
    )

    l2d_eval_loader = DataLoader(l2d_eval_full, shuffle=False, drop_last=False, **common_loader_kwargs)
    nup_eval_loader = DataLoader(nup_eval_full, shuffle=False, drop_last=False, **common_loader_kwargs)

    # ---- Instantiate encoders + projection head
    l2d_side_dim = getattr(l2d_train_full, "side_info_dim", 0) if l2d_side_information_path else 0
    nup_side_dim = 0

    l2d_enc = HeteroGraphAutoencoder(
        metadata=l2d_train_full.get_metadata(),
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        quantizer_spec=l2d_quant.spec(),
        feat_emb_dim=16,
        use_feature_mask=False,
        feature_entropy=None,
        trainable_gates=False,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        activation=args.activation,
        dropout_rate=args.dropout_rate,
        side_info_dim=l2d_side_dim,
    ).to(device)

    nup_enc = HeteroGraphAutoencoder(
        metadata=nup_train_full.get_metadata(),
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        quantizer_spec=nup_quant.spec(),
        feat_emb_dim=16,
        use_feature_mask=False,
        feature_entropy=None,
        trainable_gates=False,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        activation=args.activation,
        dropout_rate=args.dropout_rate,
        side_info_dim=nup_side_dim,
    ).to(device)

    # Infer graph embedding dim 
    graph_emb_dim = infer_graph_emb_dim(
        nup_enc, nup_quant, nup_train_loader_drop, nup_train_full.get_metadata(), embed_dim_per_type=args.embed_dim
    )
    print(f"[info] inferred graph_emb_dim = {graph_emb_dim}")

    proj_head = ProjectionHead(in_dim=graph_emb_dim, proj_dim=args.proj_dim).to(device)

    # ---- encoder checkpoint path
    if os.path.basename(args.encoder_ckpt_path) == "best_model.pt":
        side_tag = "_with_side_info" if l2d_side_information_path else "_no_side_info"
        args.encoder_ckpt_path = f"./models/graph_encoder/4B{side_tag}_best_model.pt"

    # ---- risk checkpoint path
    pred_tag = "_class" if args.prediction_mode == "classification" else "_reg"
    if os.path.basename(args.risk_ckpt_path) == "best_model.pt":
        side_tag = "_with_side_info" if l2d_side_information_path else "_no_side_info"
        args.risk_ckpt_path = f"./models/risk_from_embeddings/4C{side_tag}{pred_tag}_best_model.pt"

    # ------------------------- Stage A: train encoders + projection -------------------------
    if args.train_encoder:
        print("\n=== Stage A: TRAIN ENCODERS + PROJECTION ===")
        enc_opt = torch.optim.Adam(
            list(l2d_enc.parameters()) + list(nup_enc.parameters()) + list(proj_head.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        best_val = float("inf")

        for epoch in range(1, args.num_epochs + 1):
            l2d_enc.train()
            nup_enc.train()
            proj_head.train()

            total = 0.0
            n_batches = 0

            for l2d_batch, nup_batch in tqdm(
                zip(l2d_train_loader_drop, nup_train_loader_drop),
                total=min(len(l2d_train_loader_drop), len(nup_train_loader_drop)),
                desc=f"Epoch {epoch:02d} [train_enc]",
            ):
                enc_opt.zero_grad()

                l2d_batch = l2d_quant.transform_inplace(l2d_batch).to(device)
                nup_batch = nup_quant.transform_inplace(nup_batch).to(device)

                l2d_z, l2d_feat_logits, l2d_edge_logits = l2d_enc(l2d_batch)
                nup_z, nup_feat_logits, nup_edge_logits = nup_enc(nup_batch)

                l2d_recon = feature_loss(l2d_feat_logits, l2d_batch) + edge_loss(l2d_edge_logits, l2d_z, l2d_enc.edge_decoders)
                nup_recon = feature_loss(nup_feat_logits, nup_batch) + edge_loss(nup_edge_logits, nup_z, nup_enc.edge_decoders)

                l2d_g = batched_graph_embeddings(l2d_z, l2d_batch, l2d_train_full.get_metadata(), embed_dim_per_type=args.embed_dim)
                nup_g = batched_graph_embeddings(nup_z, nup_batch, nup_train_full.get_metadata(), embed_dim_per_type=args.embed_dim)

                l2d_p = proj_head(l2d_g)
                nup_p = proj_head(nup_g)
                kl = kl_divergence_between_gaussians(l2d_p, nup_p)

                loss = l2d_recon + nup_recon + args.kl_weight * kl
                loss.backward()
                enc_opt.step()

                total += float(loss.item())
                n_batches += 1

            train_loss = total / max(n_batches, 1)

            # ---- validation
            l2d_enc.eval()
            nup_enc.eval()
            proj_head.eval()

            v_total = 0.0
            v_kl = 0.0
            v_l2d_recon = 0.0
            v_nup_recon = 0.0
            v_batches = 0

            with torch.no_grad():
                for l2d_batch, nup_batch in tqdm(
                    zip(l2d_val_loader_drop, nup_val_loader_drop),
                    total=min(len(l2d_val_loader_drop), len(nup_val_loader_drop)),
                    desc=f"Epoch {epoch:02d} [val_enc]",
                ):
                    l2d_batch = l2d_quant.transform_inplace(l2d_batch).to(device)
                    nup_batch = nup_quant.transform_inplace(nup_batch).to(device)

                    l2d_z, l2d_feat_logits, l2d_edge_logits = l2d_enc(l2d_batch)
                    nup_z, nup_feat_logits, nup_edge_logits = nup_enc(nup_batch)

                    l2d_recon = feature_loss(l2d_feat_logits, l2d_batch) + edge_loss(l2d_edge_logits, l2d_z, l2d_enc.edge_decoders)
                    nup_recon = feature_loss(nup_feat_logits, nup_batch) + edge_loss(nup_edge_logits, nup_z, nup_enc.edge_decoders)

                    l2d_g = batched_graph_embeddings(l2d_z, l2d_batch, l2d_train_full.get_metadata(), embed_dim_per_type=args.embed_dim)
                    nup_g = batched_graph_embeddings(nup_z, nup_batch, nup_train_full.get_metadata(), embed_dim_per_type=args.embed_dim)

                    l2d_p = proj_head(l2d_g)
                    nup_p = proj_head(nup_g)
                    kl = kl_divergence_between_gaussians(l2d_p, nup_p)

                    loss = l2d_recon + nup_recon + args.kl_weight * kl

                    v_total += float(loss.item())
                    v_kl += float(kl.item())
                    v_l2d_recon += float(l2d_recon.item())
                    v_nup_recon += float(nup_recon.item())
                    v_batches += 1

            val_loss = v_total / max(v_batches, 1)
            val_kl = v_kl / max(v_batches, 1)
            val_l2d_recon = v_l2d_recon / max(v_batches, 1)
            val_nup_recon = v_nup_recon / max(v_batches, 1)

            print(
                f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
                f"| kl={val_kl:.4f} | l2d_recon={val_l2d_recon:.4f} | nuplan_recon={val_nup_recon:.4f}"
            )

            if wandb_run is not None:
                wandb.log({
                    "enc/epoch": epoch,
                    "enc/train_loss": train_loss,
                    "enc/val_loss": val_loss,
                    "enc/val_kl": val_kl,
                    "enc/val_l2d_recon": val_l2d_recon,
                    "enc/val_nuplan_recon": val_nup_recon,
                })

            if val_loss < best_val:
                best_val = val_loss
                os.makedirs(os.path.dirname(args.encoder_ckpt_path), exist_ok=True)
                torch.save(
                    {
                        "l2d_encoder": l2d_enc.state_dict(),
                        "nuplan_encoder": nup_enc.state_dict(),
                        "projection_head": proj_head.state_dict(),
                        "graph_emb_dim": graph_emb_dim,
                        "args": dict(vars(args)),
                    },
                    args.encoder_ckpt_path,
                )
                print(f"  -> saved best encoder ckpt: {args.encoder_ckpt_path}")

    # ------------------------- Load encoder ckpt if needed -------------------------
    if args.extract_embeddings or args.train_risk or args.evaluate_risk:
        _require_file(args.encoder_ckpt_path, "Encoder checkpoint (--encoder_ckpt_path)")
        state = torch.load(args.encoder_ckpt_path, map_location=device, weights_only=False)
        l2d_enc.load_state_dict(state["l2d_encoder"])
        nup_enc.load_state_dict(state["nuplan_encoder"])
        proj_head.load_state_dict(state["projection_head"])
        l2d_enc.eval()
        nup_enc.eval()
        proj_head.eval()

    # ------------------------- Stage B: extract embeddings -------------------------
    def extract_and_save_embeddings(
        dataset_name: str,
        loader: DataLoader,
        encoder: HeteroGraphAutoencoder,
        quantizer: QuantileFeatureQuantizer,
        metadata,
        split_name: str,
        out_dir: str,
    ) -> str:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{dataset_name.lower()}_{split_name}_proj_embeddings.pt")

        embs: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Extract [{dataset_name}:{split_name}]"):
                batch = quantizer.transform_inplace(batch).to(device)
                z, _, _ = encoder(batch)
                g = batched_graph_embeddings(z, batch, metadata, embed_dim_per_type=args.embed_dim)
                p = proj_head(g)

                ids = episode_ids_from_batch(batch)
                for i, eid in enumerate(ids):
                    embs[eid] = p[i].detach().cpu()

        torch.save(embs, out_path)
        print(f"[ok] saved {dataset_name} {split_name} embeddings -> {out_path}")
        return out_path

    if args.extract_embeddings:
        print("\n=== Stage B: EXTRACT EMBEDDINGS ===")
        if args.embed_split in ("train", "both"):
            extract_and_save_embeddings("L2D", l2d_train_loader_full, l2d_enc, l2d_quant, l2d_train_full.get_metadata(), "train", args.embedding_dir)
            extract_and_save_embeddings("NuPlan", nup_train_loader_full, nup_enc, nup_quant, nup_train_full.get_metadata(), "train", args.embedding_dir)
        if args.embed_split in ("eval", "both"):
            extract_and_save_embeddings("L2D", l2d_eval_loader, l2d_enc, l2d_quant, l2d_train_full.get_metadata(), "eval", args.embedding_dir)
            extract_and_save_embeddings("NuPlan", nup_eval_loader, nup_enc, nup_quant, nup_train_full.get_metadata(), "eval", args.embedding_dir)

    # ------------------------- Stage C/D: risk head from embeddings -------------------------
    def load_embeddings(path):
        _require_file(path, f"Embeddings file")
        obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, dict) or len(obj) == 0:
            raise ValueError(f"Embeddings file looks wrong/empty: {path}")
        return obj

    def build_embedding_dataset(embs, risk_json_path, prediction_mode):
        """
        Returns:
          X: [N, D] float
          y: [N, 1] float (regression) OR [N] long (classification)
        Only keeps IDs present in both embs and risk dict.
        """
        _require_file(risk_json_path, "Risk scores JSON")
        with open(risk_json_path, "r") as f:
            risk = json.load(f)

        keys = sorted(set(embs.keys()) & set(risk.keys()))
        if len(keys) == 0:
            raise RuntimeError(
                f"No overlapping episode IDs between embeddings and risk scores.\n"
                f"  embeddings: {len(embs)} ids\n"
                f"  risk json:  {len(risk)} ids\n"
                f"  risk file:  {risk_json_path}"
            )

        X = torch.stack([embs[k].float() for k in keys], dim=0)

        if prediction_mode == "regression":
            y = torch.tensor([float(risk[k]) for k in keys], dtype=torch.float32).view(-1, 1)
        else:
            # risk json may be floats; we bin using thresholds
            raw = torch.tensor([float(risk[k]) for k in keys], dtype=torch.float32)
            y = risk_to_class_safe(raw).long()

        return X, y

    class TensorDataset(torch.utils.data.Dataset):
        def __init__(self, X: torch.Tensor, y: torch.Tensor):
            self.X = X
            self.y = y
        def __len__(self) -> int:
            return self.X.shape[0]
        def __getitem__(self, idx: int):
            return self.X[idx], self.y[idx]

    # Build filenames
    nup_train_emb_path = os.path.join(args.embedding_dir, "nuplan_train_proj_embeddings.pt")
    l2d_eval_emb_path = os.path.join(args.embedding_dir, "l2d_eval_proj_embeddings.pt")
    nup_eval_emb_path = os.path.join(args.embedding_dir, "nuplan_eval_proj_embeddings.pt")

    # Risk head model init (lazy)
    risk_head: Optional[RiskPredictionHead] = None

    if args.train_risk or args.evaluate_risk:
        # Need embeddings present
        _require_file(nup_train_emb_path, "NuPlan TRAIN embeddings (expected from --extract_embeddings --embed_split train/both)")
        _require_file(nup_eval_emb_path, "NuPlan EVAL embeddings (expected from --extract_embeddings --embed_split eval/both)")
        _require_file(l2d_eval_emb_path, "L2D EVAL embeddings (expected from --extract_embeddings --embed_split eval/both)")

    if args.train_risk:
        print("\n=== Stage C: TRAIN RISK HEAD (from embeddings) ===")

        nup_train_embs = load_embeddings(nup_train_emb_path)

        X, y = build_embedding_dataset(nup_train_embs, nup_paths["train_risk_path"], args.prediction_mode)
        ds = TensorDataset(X, y)

        split_gen2 = torch.Generator().manual_seed(args.seed)
        train_size = int((1 - args.val_fraction) * len(ds))
        val_size = len(ds) - train_size
        ds_train, ds_val = random_split(ds, [train_size, val_size], generator=split_gen2)

        tr_loader = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
        va_loader = torch.utils.data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)

        in_dim = X.shape[1]
        out_dim = 1 if args.prediction_mode == "regression" else args.num_classes
        risk_head = RiskPredictionHead(
            input_dim=in_dim,
            hidden_dim=args.risk_hidden_dim,
            output_dim=out_dim,
            mode=args.prediction_mode,
        ).to(device)

        loss_fn = nn.MSELoss() if args.prediction_mode == "regression" else nn.CrossEntropyLoss()
        opt = torch.optim.Adam(risk_head.parameters(), lr=args.risk_lr, weight_decay=args.risk_weight_decay)

        best = float("inf")
        for epoch in range(1, args.risk_epochs + 1):
            risk_head.train()
            total = 0.0
            correct = 0
            seen = 0

            for xb, yb in tqdm(tr_loader, desc=f"Epoch {epoch:02d} [train_risk]"):
                xb = xb.to(device)
                yb = yb.to(device)

                opt.zero_grad()
                pred = risk_head(xb)

                if args.prediction_mode == "regression":
                    loss = loss_fn(pred, yb)
                else:
                    loss = loss_fn(pred, yb.long())
                    pred_cls = pred.argmax(dim=-1)
                    correct += int((pred_cls == yb).sum().item())
                    seen += int(yb.numel())

                loss.backward()
                opt.step()
                total += float(loss.item())

            train_loss = total / max(len(tr_loader), 1)
            train_acc = (correct / max(seen, 1)) if args.prediction_mode == "classification" else None

            # val
            risk_head.eval()
            vtotal = 0.0
            vcorrect = 0
            vseen = 0
            with torch.no_grad():
                for xb, yb in tqdm(va_loader, desc=f"Epoch {epoch:02d} [val_risk]"):
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = risk_head(xb)

                    if args.prediction_mode == "regression":
                        vtotal += float(loss_fn(pred, yb).item())
                    else:
                        vtotal += float(loss_fn(pred, yb.long()).item())
                        pred_cls = pred.argmax(dim=-1)
                        vcorrect += int((pred_cls == yb).sum().item())
                        vseen += int(yb.numel())

            val_loss = vtotal / max(len(va_loader), 1)
            val_acc = (vcorrect / max(vseen, 1)) if args.prediction_mode == "classification" else None

            if args.prediction_mode == "classification":
                print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")
            else:
                print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

            if wandb_run is not None:
                payload = {
                    "risk/epoch": epoch,
                    "risk/train_loss": train_loss,
                    "risk/val_loss": val_loss,
                }
                if args.prediction_mode == "classification":
                    payload.update({"risk/train_acc": train_acc, "risk/val_acc": val_acc})
                wandb.log(payload)

            if val_loss < best:
                best = val_loss
                os.makedirs(os.path.dirname(args.risk_ckpt_path), exist_ok=True)
                torch.save(
                    {
                        "state_dict": risk_head.state_dict(),
                        "in_dim": in_dim,
                        "out_dim": out_dim,
                        "args": dict(vars(args)),
                    },
                    args.risk_ckpt_path,
                )
                print(f"  -> saved best risk ckpt: {args.risk_ckpt_path}")

    if args.evaluate_risk:
        print("\n=== Stage D: EVALUATE RISK HEAD (on EVAL embeddings) ===")

        _require_file(args.risk_ckpt_path, "Risk checkpoint (--risk_ckpt_path)")
        ck = torch.load(args.risk_ckpt_path, map_location=device, weights_only=False)

        in_dim = int(ck["in_dim"])
        out_dim = int(ck["out_dim"])
        risk_head = RiskPredictionHead(
            input_dim=in_dim,
            hidden_dim=args.risk_hidden_dim,
            output_dim=out_dim,
            mode=args.prediction_mode,
        ).to(device)
        risk_head.load_state_dict(ck["state_dict"])
        risk_head.eval()

        loss_fn = nn.MSELoss() if args.prediction_mode == "regression" else nn.CrossEntropyLoss()

        # Load EVAL embeddings
        nup_eval_embs = load_embeddings(nup_eval_emb_path)
        l2d_eval_embs = load_embeddings(l2d_eval_emb_path)

        # Build eval tensors (labels from EVAL risk files)
        Xn, yn = build_embedding_dataset(nup_eval_embs, nup_paths["eval_risk_path"], args.prediction_mode)
        Xl, yl = build_embedding_dataset(l2d_eval_embs, l2d_paths["eval_risk_path"], args.prediction_mode)

        def eval_tensor(X: torch.Tensor, y: torch.Tensor, name: str) -> Dict[str, Any]:
            dl = torch.utils.data.DataLoader(TensorDataset(X, y), batch_size=args.batch_size, shuffle=False)
            total = 0.0
            correct = 0
            seen = 0
            all_true = []
            all_pred = []

            with torch.no_grad():
                for xb, yb in tqdm(dl, desc=f"[eval_risk:{name}]"):
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = risk_head(xb)

                    if args.prediction_mode == "regression":
                        total += float(loss_fn(pred, yb).item())
                    else:
                        total += float(loss_fn(pred, yb.long()).item())
                        pred_cls = pred.argmax(dim=-1)
                        correct += int((pred_cls == yb).sum().item())
                        seen += int(yb.numel())
                        all_true.append(yb.detach().cpu())
                        all_pred.append(pred_cls.detach().cpu())

            avg_loss = total / max(len(dl), 1)

            if args.prediction_mode == "classification":
                acc = correct / max(seen, 1)
                y_true = torch.cat(all_true, dim=0) if all_true else torch.empty(0, dtype=torch.long)
                y_pred = torch.cat(all_pred, dim=0) if all_pred else torch.empty(0, dtype=torch.long)
                cm = compute_confusion_matrix(args.num_classes, y_true, y_pred).cpu()
                return {"loss": avg_loss, "acc": acc, "cm": cm}
            return {"loss": avg_loss}

        nu = eval_tensor(Xn, yn, "NuPlan_EVAL")
        l2 = eval_tensor(Xl, yl, "L2D_EVAL")

        if args.prediction_mode == "classification":
            print(f"NuPlan EVAL | loss={nu['loss']:.4f} | acc={nu['acc']:.4f}")
            print("NuPlan confusion matrix (rows=true, cols=pred):\n", nu["cm"].numpy())

            print(f"L2D   EVAL | loss={l2['loss']:.4f} | acc={l2['acc']:.4f}")
            print("L2D confusion matrix (rows=true, cols=pred):\n", l2["cm"].numpy())
        else:
            print(f"NuPlan EVAL | mse={nu['loss']:.6f}")
            print(f"L2D   EVAL | mse={l2['loss']:.6f}")

        if wandb_run is not None:
            payload = {
                "risk_eval/nuplan_loss": float(nu["loss"]),
                "risk_eval/l2d_loss": float(l2["loss"]),
            }
            if args.prediction_mode == "classification":
                payload.update({
                    "risk_eval/nuplan_acc": float(nu["acc"]),
                    "risk_eval/l2d_acc": float(l2["acc"]),
                })
                # Log confusion matrices as tables (W&B doesn't have a universal native CM object)
                nu_cm = nu["cm"].numpy().tolist()
                l2_cm = l2["cm"].numpy().tolist()
                payload["risk_eval/nuplan_confusion_matrix"] = wandb.Table(
                    data=nu_cm,
                    columns=[f"pred_{i}" for i in range(args.num_classes)],
                )
                payload["risk_eval/l2d_confusion_matrix"] = wandb.Table(
                    data=l2_cm,
                    columns=[f"pred_{i}" for i in range(args.num_classes)],
                )
            wandb.log(payload)

    if wandb_run is not None:
        wandb_run.finish()


# ------------------------- CLI -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified: encoder+projection + risk head from embeddings.")

    # Data root (new structure)
    parser.add_argument("--data_root", type=str, default="data")

    # Encoder stage flags
    parser.add_argument("--train_encoder", action="store_true")
    parser.add_argument("--mode", type=str, default="all")
    parser.add_argument("--side_info_path", type=str, default=None)
    parser.add_argument("--node_features_to_exclude", nargs="+", type=str, default=None)

    # Embedding extraction
    parser.add_argument("--extract_embeddings", action="store_true")
    parser.add_argument("--embed_split", type=str, default="both", choices=["train", "eval", "both"])
    parser.add_argument("--embedding_dir", type=str, default="./data/graph_embeddings/")

    # Risk head stage flags
    parser.add_argument("--train_risk", action="store_true")
    parser.add_argument("--evaluate_risk", action="store_true")

    # Encoder model hyperparams
    parser.add_argument("--quant_bins", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--proj_dim", type=int, default=32)
    parser.add_argument("--num_encoder_layers", type=int, default=1)
    parser.add_argument("--num_decoder_layers", type=int, default=1)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--dropout_rate", type=float, default=0.1)

    # Encoder optimizer
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--kl_weight", type=float, default=0.1)

    # Risk head hyperparams
    parser.add_argument("--prediction_mode", type=str, default="regression", choices=["regression", "classification"])
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--risk_hidden_dim", type=int, default=64)
    parser.add_argument("--risk_lr", type=float, default=1e-4)
    parser.add_argument("--risk_weight_decay", type=float, default=1e-5)
    parser.add_argument("--risk_epochs", type=int, default=10)

    # Classification binning thresholds
    parser.add_argument(
        "--class_thresholds",
        type=float,
        nargs=3,
        default=(0.25, 0.50, 0.75),
        help="Three thresholds t1 t2 t3 defining 4 bins.",
    )

    # Loader + splitting
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=5)

    # Paths / checkpoints
    parser.add_argument("--encoder_ckpt_path", type=str, default="./models/graph_encoder/best_model.pt")
    parser.add_argument("--risk_ckpt_path", type=str, default="./models/risk_from_embeddings/best_model.pt")

    # Misc
    parser.add_argument("--seed", type=int, default=42)

    # wandb
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    # Config file
    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args()
    args = apply_yaml_overrides(parser, args)

    # Run
    run(args)
