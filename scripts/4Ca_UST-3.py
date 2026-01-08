import argparse
import os
import sys
import json
from typing import Dict, Any, List, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    mean_absolute_error,
    cohen_kappa_score,
    mean_squared_error,
    r2_score,
    confusion_matrix,
)
from scipy.stats import spearmanr

import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_encoding.data_loaders import get_graph_dataset
from src.graph_encoding.autoencoder import (
    HeteroGraphAutoencoder,
    feature_loss,
    edge_loss,
    QuantileFeatureQuantizer,
    ProjectionHead,
    batched_graph_embeddings,
)
from src.graph_encoding.risk_prediction import RiskPredictionHead
from src.experiment_utils.C_utils import (
    set_seed,
    seed_worker,
    _require_dir,
    _require_file,
    _print_access,
    episode_ids_from_batch,
    risk_to_class_safe,
    apply_yaml_overrides,
    resolve_paths,
    infer_graph_emb_dim,
    _make_fallback_run_id,
    _ensure_dir,
    _running_under_wandb_agent_or_sweep,
    _str2bool,
    effective_weight,
    _torch_load_compat,
    _load_single_encoder_ckpt_flexible,
    _override_model_args_from_encoder_ckpt,
    _preload_eval_encoder_args,
    semantic_sim_cross,
    cosine_sim_cross,
    tag_weighted_teacher_probs,
    classification_metrics_from_cm,
    _masked_alignment_mse,
    _format_confusion_matrix,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Main
# -------------------------
def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    # -------------------------
    # W&B init
    # -------------------------
    wandb_run = None
    global_step = 0

    if _running_under_wandb_agent_or_sweep() and (not args.use_wandb) and (args.wandb_mode != "disabled"):
        args.use_wandb = True

    def wb_log(payload: Dict[str, Any], bump: bool = False):
        nonlocal global_step
        if (wandb_run is None) and (wandb.run is None):
            return
        if bump:
            global_step += 1
        payload = dict(payload)
        payload["global_step"] = global_step
        wandb.log(payload)

    wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT")

    if args.use_wandb and args.wandb_mode != "disabled":
        wandb_run = wandb.init(
            project=wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name or None,
            group=args.wandb_group or None,
            mode=args.wandb_mode,
            config=vars(args),
        )
        for k, v in wandb.config.items():
            if hasattr(args, k):
                setattr(args, k, v)

        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")

    # -------------------------
    # Run isolation / naming
    # -------------------------
    is_eval_only = bool(args.evaluate_risk) and not bool(args.train_encoders_recon) and not bool(args.train_proj_risk_jointly)

    if is_eval_only:
        proj_risk_ckpt_abs = os.path.abspath(args.proj_risk_ckpt_path)
        _require_file(proj_risk_ckpt_abs, "Projection/Risk checkpoint (--proj_risk_ckpt_path) for eval-only")
        run_dir = os.path.dirname(proj_risk_ckpt_abs)
        args.proj_risk_ckpt_path = proj_risk_ckpt_abs

        if args.encoder_ckpt_path is not None:
            args.encoder_ckpt_path = os.path.abspath(args.encoder_ckpt_path)

        run_id = args.run_id or (wandb_run.id if wandb_run is not None else _make_fallback_run_id())
        if wandb_run is not None:
            wandb.config.update({"run_id": run_id, "run_dir": run_dir}, allow_val_change=True)
    else:
        run_id = args.run_id or (wandb_run.id if wandb_run is not None else _make_fallback_run_id())
        pred_dir = "classification" if args.prediction_mode == "classification" else "regression"
        output_root = _ensure_dir(os.path.abspath(args.output_root))
        run_dir = _ensure_dir(os.path.join(output_root, "4Ca", "L2D_NuPlan", pred_dir, run_id))

        if os.path.basename(args.encoder_ckpt_path) == "best_model.pt":
            args.encoder_ckpt_path = os.path.join(run_dir, "4Ca_L2D_NuPlan_enc_best_model.pt")
        else:
            args.encoder_ckpt_path = os.path.abspath(args.encoder_ckpt_path)

        if os.path.basename(args.proj_risk_ckpt_path) == "best_model.pt":
            args.proj_risk_ckpt_path = os.path.join(run_dir, "4Ca_L2D_NuPlan_proj_risk_best_model.pt")
        else:
            args.proj_risk_ckpt_path = os.path.abspath(args.proj_risk_ckpt_path)

        if wandb_run is not None:
            wandb.config.update({"run_id": run_id, "run_dir": run_dir}, allow_val_change=True)

    # -------------------------
    # Resolve dataset paths
    # -------------------------
    args.data_root = os.path.abspath(args.data_root)
    l2d_paths = resolve_paths(args.data_root, "L2D")
    nup_paths = resolve_paths(args.data_root, "NuPlan")

    _print_access("L2D TRAIN graphs", l2d_paths["train_graph_root"])
    _print_access("L2D TRAIN risk", l2d_paths["train_risk_path"])
    _print_access("L2D EVAL  graphs", l2d_paths["eval_graph_root"])
    _print_access("L2D EVAL  risk", l2d_paths["eval_risk_path"])

    _print_access("NuPlan TRAIN graphs", nup_paths["train_graph_root"])
    _print_access("NuPlan TRAIN risk", nup_paths["train_risk_path"])
    _print_access("NuPlan EVAL  graphs", nup_paths["eval_graph_root"])
    _print_access("NuPlan EVAL  risk", nup_paths["eval_risk_path"])

    # Side info path (only used for L2D if provided)
    l2d_side_information_path = None
    if args.side_info_path is not None:
        l2d_side_information_path = os.path.abspath(args.side_info_path)
        _print_access("L2D side info", l2d_side_information_path)

    unified_tags_path = os.path.abspath(args.unified_tags_path)
    _print_access("Unified tags", unified_tags_path)
    with open(unified_tags_path, "r") as f:
        unified_tags_data = json.load(f)

    episode_to_tags: Dict[str, Set[str]] = {}
    for unified_episode_key, tags in unified_tags_data.items():
        if unified_episode_key.startswith("L2D_"):
            key_for_lookup = unified_episode_key.replace("L2D_", "")
        elif unified_episode_key.startswith("NuPlan_"):
            original_filename_base = unified_episode_key.replace("NuPlan_", "")
            key_for_lookup = original_filename_base.split("_")[0]
        else:
            key_for_lookup = unified_episode_key

        episode_to_tags[key_for_lookup] = set(tags)
    print(f"Loaded tags for {len(episode_to_tags)} episodes.")

    # -------------------------
    # Datasets
    # -------------------------
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

    # Quantizers
    l2d_quant = QuantileFeatureQuantizer(bins=args.quant_bins, node_types=l2d_train_full.get_metadata()[0])
    nup_quant = QuantileFeatureQuantizer(bins=args.quant_bins, node_types=nup_train_full.get_metadata()[0])
    print("Fitting quantizers on TRAIN datasets...")
    l2d_quant.fit(l2d_train_full)
    nup_quant.fit(nup_train_full)

    # Splits
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

    l2d_train_loader_drop = DataLoader(l2d_train_ds, shuffle=True, drop_last=True, **common_loader_kwargs)
    nup_train_loader_drop = DataLoader(nup_train_ds, shuffle=True, drop_last=True, **common_loader_kwargs)
    l2d_val_loader_drop = DataLoader(l2d_val_ds, shuffle=False, drop_last=True, **common_loader_kwargs)
    nup_val_loader_drop = DataLoader(nup_val_ds, shuffle=False, drop_last=True, **common_loader_kwargs)

    # Eval datasets/loaders
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

    # override encoder-shape args from ckpts 
    _preload_eval_encoder_args(args)

    l2d_side_dim = getattr(l2d_train_full, "side_info_dim", 0) if l2d_side_information_path else 0
    nup_side_dim = 0

    # -------------------------
    # Build encoders
    # -------------------------
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

    # Infer graph_emb_dim from current encoder build (NuPlan reference)
    graph_emb_dim = infer_graph_emb_dim(
        nup_enc,
        nup_quant,
        nup_train_loader_drop,
        nup_train_full.get_metadata(),
        embed_dim_per_type=args.embed_dim,
    )
    print(f"[info] inferred graph_emb_dim = {graph_emb_dim}")

    proj_head = ProjectionHead(in_dim=graph_emb_dim, proj_dim=args.proj_dim).to(device)

    # -------------------------
    # Optional: load pretrained 4B encoders
    # -------------------------
    if args.load_pretrained_4b_encoders:
        _require_file(args.l2d_4b_ckpt_path, "--l2d_4b_ckpt_path required with --load_pretrained_4b_encoders")
        _require_file(args.nup_4b_ckpt_path, "--nup_4b_ckpt_path required with --load_pretrained_4b_encoders")
        print("[info] Loading pretrained 4B encoder state_dicts...")
        _load_single_encoder_ckpt_flexible(l2d_enc, args.l2d_4b_ckpt_path)
        _load_single_encoder_ckpt_flexible(nup_enc, args.nup_4b_ckpt_path)
        l2d_enc.eval()
        nup_enc.eval()

    # ------------------------- Stage A1: TRAIN ENCODERS (RECON ONLY) -------------------------
    if args.train_encoders_recon:
        print("\n=== Stage A1: TRAIN ENCODERS (RECON ONLY) ===")
        if args.load_pretrained_4b_encoders and not args.force_retrain_encoders:
            print("[info] Skipping A1 because --load_pretrained_4b_encoders is set (use --force_retrain_encoders to retrain).")
        else:
            enc_opt = torch.optim.Adam(
                list(l2d_enc.parameters()) + list(nup_enc.parameters()),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            best_val = float("inf")

            for epoch in range(1, args.num_epochs + 1):
                l2d_enc.train()
                nup_enc.train()

                total = 0.0
                l2d_recon_total = 0.0
                nup_recon_total = 0.0
                n_batches = 0

                for l2d_batch, nup_batch in tqdm(
                    zip(l2d_train_loader_drop, nup_train_loader_drop),
                    total=min(len(l2d_train_loader_drop), len(nup_train_loader_drop)),
                    desc=f"Epoch {epoch:02d} [train_enc_recon]",
                ):
                    enc_opt.zero_grad()

                    l2d_batch = l2d_quant.transform_inplace(l2d_batch).to(device)
                    nup_batch = nup_quant.transform_inplace(nup_batch).to(device)

                    l2d_z, l2d_feat_logits, l2d_edge_logits = l2d_enc(l2d_batch)
                    nup_z, nup_feat_logits, nup_edge_logits = nup_enc(nup_batch)

                    l2d_recon = feature_loss(l2d_feat_logits, l2d_batch) + edge_loss(
                        l2d_edge_logits, l2d_z, l2d_enc.edge_decoders
                    )
                    nup_recon = feature_loss(nup_feat_logits, nup_batch) + edge_loss(
                        nup_edge_logits, nup_z, nup_enc.edge_decoders
                    )

                    loss = l2d_recon + nup_recon
                    loss.backward()
                    enc_opt.step()

                    total += float(loss.item())
                    l2d_recon_total += float(l2d_recon.item())
                    nup_recon_total += float(nup_recon.item())
                    n_batches += 1

                train_loss = total / max(n_batches, 1)
                train_l2d_recon = l2d_recon_total / max(n_batches, 1)
                train_nup_recon = nup_recon_total / max(n_batches, 1)

                # val
                l2d_enc.eval()
                nup_enc.eval()
                v_total = 0.0
                v_l2d_recon = 0.0
                v_nup_recon = 0.0
                v_batches = 0

                with torch.no_grad():
                    for l2d_batch, nup_batch in tqdm(
                        zip(l2d_val_loader_drop, nup_val_loader_drop),
                        total=min(len(l2d_val_loader_drop), len(nup_val_loader_drop)),
                        desc=f"Epoch {epoch:02d} [val_enc_recon]",
                    ):
                        l2d_batch = l2d_quant.transform_inplace(l2d_batch).to(device)
                        nup_batch = nup_quant.transform_inplace(nup_batch).to(device)

                        l2d_z, l2d_feat_logits, l2d_edge_logits = l2d_enc(l2d_batch)
                        nup_z, nup_feat_logits, nup_edge_logits = nup_enc(nup_batch)

                        l2d_recon = feature_loss(l2d_feat_logits, l2d_batch) + edge_loss(
                            l2d_edge_logits, l2d_z, l2d_enc.edge_decoders
                        )
                        nup_recon = feature_loss(nup_feat_logits, nup_batch) + edge_loss(
                            nup_edge_logits, nup_z, nup_enc.edge_decoders
                        )

                        loss = l2d_recon + nup_recon
                        v_total += float(loss.item())
                        v_l2d_recon += float(l2d_recon.item())
                        v_nup_recon += float(nup_recon.item())
                        v_batches += 1

                val_loss = v_total / max(v_batches, 1)
                val_l2d_recon = v_l2d_recon / max(v_batches, 1)
                val_nup_recon = v_nup_recon / max(v_batches, 1)

                print(
                    f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
                    f"| val_l2d_recon={val_l2d_recon:.4f} | val_nuplan_recon={val_nup_recon:.4f}"
                )

                wb_log(
                    {
                        "enc_recon/epoch": epoch,
                        "enc_recon/train_loss": train_loss,
                        "enc_recon/train_l2d_recon": train_l2d_recon,
                        "enc_recon/train_nuplan_recon": train_nup_recon,
                        "enc_recon/val_loss": val_loss,
                        "enc_recon/val_l2d_recon": val_l2d_recon,
                        "enc_recon/val_nuplan_recon": val_nup_recon,
                    },
                    bump=True,
                )

                if val_loss < best_val:
                    best_val = val_loss
                    os.makedirs(os.path.dirname(args.encoder_ckpt_path), exist_ok=True)
                    torch.save(
                        {
                            "l2d_encoder": l2d_enc.state_dict(),
                            "nuplan_encoder": nup_enc.state_dict(),
                            "projection_head": proj_head.state_dict(),
                            "graph_emb_dim": graph_emb_dim,
                            "best_val_loss": best_val,
                            "stage": "A1_recon",
                            "args": dict(vars(args)),
                            "run_dir": run_dir,
                        },
                        args.encoder_ckpt_path,
                    )
                    print(f"  -> saved best A1 encoder ckpt: {args.encoder_ckpt_path}")

    # ------------------------- Stage 2: TRAIN PROJECTION + RISK (ALIGN + DISTILL ADDED) -------------------------
    if args.train_proj_risk_jointly:
        print("\n=== Stage 2: TRAIN PROJECTION + RISK (ALIGN + DISTILL ADDED, NOT REPLACED) ===")

        if (not args.load_pretrained_4b_encoders) and (not args.train_encoders_recon):
            _require_file(args.encoder_ckpt_path, "Encoder checkpoint for Stage 2 (--encoder_ckpt_path)")
            state = _torch_load_compat(args.encoder_ckpt_path, map_location=device)
            _override_model_args_from_encoder_ckpt(args, state)
            l2d_enc.load_state_dict(state["l2d_encoder"])
            nup_enc.load_state_dict(state["nuplan_encoder"])

        l2d_enc.eval()
        nup_enc.eval()
        for p in l2d_enc.parameters():
            p.requires_grad = False
        for p in nup_enc.parameters():
            p.requires_grad = False

        risk_head = RiskPredictionHead(
            input_dim=args.proj_dim,
            hidden_dim=args.risk_hidden_dim,
            output_dim=1 if args.prediction_mode == "regression" else args.num_classes,
            mode=args.prediction_mode,
        ).to(device)

        risk_loss_fn = nn.MSELoss() if args.prediction_mode == "regression" else nn.CrossEntropyLoss()

        joint_opt = torch.optim.Adam(
            list(proj_head.parameters()) + list(risk_head.parameters()),
            lr=args.proj_risk_lr,
            weight_decay=args.proj_risk_weight_decay,
        )

        best_val_loss = float("inf")

        for epoch in range(1, args.proj_risk_epochs + 1):
            proj_head.train()
            risk_head.train()

            align_w = effective_weight(epoch, args.align_weight, args.align_warmup_epochs, args.align_ramp_epochs)
            distill_w = effective_weight(epoch, args.distill_weight, args.distill_warmup_epochs, args.distill_ramp_epochs)

            total_loss = 0.0
            total_align = 0.0
            total_risk = 0.0
            total_distill = 0.0
            n_batches = 0

            for l2d_batch, nup_batch in tqdm(
                zip(l2d_train_loader_drop, nup_train_loader_drop),
                total=min(len(l2d_train_loader_drop), len(nup_train_loader_drop)),
                desc=f"Epoch {epoch:02d} [train_joint]",
            ):
                joint_opt.zero_grad()

                with torch.no_grad():
                    l2d_batch = l2d_quant.transform_inplace(l2d_batch).to(device)
                    nup_batch = nup_quant.transform_inplace(nup_batch).to(device)

                    l2d_z, _, _ = l2d_enc(l2d_batch)
                    nup_z, _, _ = nup_enc(nup_batch)

                    l2d_g = batched_graph_embeddings(
                        l2d_z, l2d_batch, l2d_train_full.get_metadata(), embed_dim_per_type=args.embed_dim
                    )
                    nup_g = batched_graph_embeddings(
                        nup_z, nup_batch, nup_train_full.get_metadata(), embed_dim_per_type=args.embed_dim
                    )

                l2d_p = proj_head(l2d_g)
                nup_p = proj_head(nup_g)

                align_loss = torch.tensor(0.0, device=device)
                if align_w > 0.0:
                    tags_l2d = [episode_to_tags.get(ep_id, set()) for ep_id in episode_ids_from_batch(l2d_batch)]
                    tags_nup = [episode_to_tags.get(ep_id, set()) for ep_id in episode_ids_from_batch(nup_batch)]
                    any_tags = (sum(len(t) for t in tags_l2d) > 0) or (sum(len(t) for t in tags_nup) > 0)
                    if any_tags:
                        S = semantic_sim_cross(tags_nup, tags_l2d, device)
                        E = cosine_sim_cross(nup_p, l2d_p)
                        align_loss = _masked_alignment_mse(E, S, tags_nup, tags_l2d)

                nup_logits = risk_head(nup_p)
                if args.prediction_mode == "regression":
                    nup_target = nup_batch.y.view(-1, 1).float()
                else:
                    nup_target = risk_to_class_safe(nup_batch.y).view(-1).long()
                risk_loss = risk_loss_fn(nup_logits, nup_target)

                distill_loss = torch.tensor(0.0, device=device)
                if distill_w > 0.0:
                    if args.prediction_mode != "classification":
                        distill_loss = torch.tensor(0.0, device=device)
                    else:
                        with torch.no_grad():
                            t_probs_hq = F.softmax(nup_logits / float(args.distill_temp), dim=-1)
                            tags_l2d = [episode_to_tags.get(ep_id, set()) for ep_id in episode_ids_from_batch(l2d_batch)]
                            tags_nup = [episode_to_tags.get(ep_id, set()) for ep_id in episode_ids_from_batch(nup_batch)]
                            t_probs_lq = tag_weighted_teacher_probs(
                                t_probs_hq, tags_nup, tags_l2d, tag_softmax_temp=args.tag_softmax_temp
                            )

                        l2d_logits = risk_head(l2d_p)
                        s_logp = F.log_softmax(l2d_logits / float(args.distill_temp), dim=-1)
                        distill_loss = F.kl_div(s_logp, t_probs_lq, reduction="batchmean") * (float(args.distill_temp) ** 2)

                loss = (align_w * align_loss) + (args.risk_loss_weight * risk_loss) + (distill_w * distill_loss)
                loss.backward()
                joint_opt.step()

                total_loss += float(loss.item())
                total_align += float(align_loss.item())
                total_risk += float(risk_loss.item())
                total_distill += float(distill_loss.item())
                n_batches += 1

            train_loss = total_loss / max(n_batches, 1)
            train_align = total_align / max(n_batches, 1)
            train_risk = total_risk / max(n_batches, 1)
            train_dist = total_distill / max(n_batches, 1)

            proj_head.eval()
            risk_head.eval()
            val_total = 0.0
            val_align = 0.0
            val_risk = 0.0
            val_dist = 0.0
            v_correct = 0
            v_seen = 0
            v_batches = 0

            with torch.no_grad():
                for l2d_batch, nup_batch in tqdm(
                    zip(l2d_val_loader_drop, nup_val_loader_drop),
                    total=min(len(l2d_val_loader_drop), len(nup_val_loader_drop)),
                    desc=f"Epoch {epoch:02d} [val_joint]",
                ):
                    l2d_batch = l2d_quant.transform_inplace(l2d_batch).to(device)
                    nup_batch = nup_quant.transform_inplace(nup_batch).to(device)

                    l2d_z, _, _ = l2d_enc(l2d_batch)
                    nup_z, _, _ = nup_enc(nup_batch)

                    l2d_g = batched_graph_embeddings(
                        l2d_z, l2d_batch, l2d_train_full.get_metadata(), embed_dim_per_type=args.embed_dim
                    )
                    nup_g = batched_graph_embeddings(
                        nup_z, nup_batch, nup_train_full.get_metadata(), embed_dim_per_type=args.embed_dim
                    )

                    l2d_p = proj_head(l2d_g)
                    nup_p = proj_head(nup_g)

                    align_loss = torch.tensor(0.0, device=device)
                    if align_w > 0.0:
                        tags_l2d = [episode_to_tags.get(ep_id, set()) for ep_id in episode_ids_from_batch(l2d_batch)]
                        tags_nup = [episode_to_tags.get(ep_id, set()) for ep_id in episode_ids_from_batch(nup_batch)]
                        any_tags = (sum(len(t) for t in tags_l2d) > 0) or (sum(len(t) for t in tags_nup) > 0)
                        if any_tags:
                            S = semantic_sim_cross(tags_nup, tags_l2d, device)
                            E = cosine_sim_cross(nup_p, l2d_p)
                            align_loss = _masked_alignment_mse(E, S, tags_nup, tags_l2d)

                    nup_logits = risk_head(nup_p)
                    if args.prediction_mode == "regression":
                        nup_target = nup_batch.y.view(-1, 1).float()
                    else:
                        nup_target = risk_to_class_safe(nup_batch.y).view(-1).long()
                        pred_cls = nup_logits.argmax(dim=-1)
                        v_correct += int((pred_cls == nup_target).sum().item())
                        v_seen += int(nup_target.numel())
                    risk_loss = risk_loss_fn(nup_logits, nup_target)

                    distill_loss = torch.tensor(0.0, device=device)
                    if distill_w > 0.0 and args.prediction_mode == "classification":
                        t_probs_hq = F.softmax(nup_logits / float(args.distill_temp), dim=-1)
                        tags_l2d = [episode_to_tags.get(ep_id, set()) for ep_id in episode_ids_from_batch(l2d_batch)]
                        tags_nup = [episode_to_tags.get(ep_id, set()) for ep_id in episode_ids_from_batch(nup_batch)]
                        t_probs_lq = tag_weighted_teacher_probs(
                            t_probs_hq, tags_nup, tags_l2d, tag_softmax_temp=args.tag_softmax_temp
                        )

                        l2d_logits = risk_head(l2d_p)
                        s_logp = F.log_softmax(l2d_logits / float(args.distill_temp), dim=-1)
                        distill_loss = F.kl_div(s_logp, t_probs_lq, reduction="batchmean") * (float(args.distill_temp) ** 2)

                    loss = (align_w * align_loss) + (args.risk_loss_weight * risk_loss) + (distill_w * distill_loss)

                    val_total += float(loss.item())
                    val_align += float(align_loss.item())
                    val_risk += float(risk_loss.item())
                    val_dist += float(distill_loss.item())
                    v_batches += 1

            val_loss = val_total / max(v_batches, 1)
            val_align = val_align / max(v_batches, 1)
            val_risk = val_risk / max(v_batches, 1)
            val_dist = val_dist / max(v_batches, 1)

            log_str = (
                f"Epoch {epoch:02d} | train_loss={train_loss:.4f} (align={train_align:.4f} risk={train_risk:.4f} dist={train_dist:.4f}) "
                f"| val_loss={val_loss:.4f} (align={val_align:.4f} risk={val_risk:.4f} dist={val_dist:.4f}) "
                f"| align_w={align_w:.4f} dist_w={distill_w:.4f}"
            )
            log_payload = {
                "joint/epoch": epoch,
                "joint/train_loss": train_loss,
                "joint/train_align_loss": train_align,
                "joint/train_risk_loss": train_risk,
                "joint/train_distill_loss": train_dist,
                "joint/val_loss": val_loss,
                "joint/val_align_loss": val_align,
                "joint/val_risk_loss": val_risk,
                "joint/val_distill_loss": val_dist,
                "joint/align_w": align_w,
                "joint/distill_w": distill_w,
            }

            if args.prediction_mode == "classification":
                val_acc = v_correct / max(v_seen, 1)
                log_str += f" | val_hq_acc={val_acc:.4f}"
                log_payload["joint/val_hq_acc"] = val_acc

            print(log_str)
            wb_log(log_payload, bump=True)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(args.proj_risk_ckpt_path), exist_ok=True)
                torch.save(
                    {
                        "projection_head": proj_head.state_dict(),
                        "risk_head": risk_head.state_dict(),
                        "best_val_loss": best_val_loss,
                        "stage": "proj_risk_joint_align_plus_distill",
                        "args": dict(vars(args)),
                        "run_dir": run_dir,
                        "graph_emb_dim": graph_emb_dim,
                    },
                    args.proj_risk_ckpt_path,
                )
                print(f"  -> saved best joint proj+risk ckpt: {args.proj_risk_ckpt_path}")

    # ------------------------- Stage 3: EVALUATE RISK (END-TO-END) -------------------------
    if args.evaluate_risk:
        print("\n=== Stage 3: EVALUATE RISK (END-TO-END) ===")

        if is_eval_only:
            run_dir = os.path.dirname(os.path.abspath(args.proj_risk_ckpt_path))

        _require_file(args.l2d_4b_ckpt_path, "L2D encoder checkpoint (--l2d_4b_ckpt_path)")
        _require_file(args.nup_4b_ckpt_path, "NuPlan encoder checkpoint (--nup_4b_ckpt_path)")
        _require_file(args.proj_risk_ckpt_path, "Projection+Risk checkpoint (--proj_risk_ckpt_path)")

        _load_single_encoder_ckpt_flexible(l2d_enc, args.l2d_4b_ckpt_path)
        _load_single_encoder_ckpt_flexible(nup_enc, args.nup_4b_ckpt_path)
        l2d_enc.eval()
        nup_enc.eval()

        proj_risk_state = _torch_load_compat(args.proj_risk_ckpt_path, map_location=device)
        if not isinstance(proj_risk_state, dict):
            raise RuntimeError("Projection+Risk checkpoint must be a dict.")
        if "projection_head" not in proj_risk_state or "risk_head" not in proj_risk_state:
            raise RuntimeError(f"Projection+Risk checkpoint missing keys. Found: {list(proj_risk_state.keys())[:30]}")

        proj_head.load_state_dict(proj_risk_state["projection_head"], strict=True)
        proj_head.eval()

        risk_out = 1 if args.prediction_mode == "regression" else args.num_classes
        risk_head = RiskPredictionHead(
            input_dim=args.proj_dim,
            hidden_dim=args.risk_hidden_dim,
            output_dim=risk_out,
            mode=args.prediction_mode,
        ).to(device)
        risk_head.load_state_dict(proj_risk_state["risk_head"], strict=True)
        risk_head.eval()

        risk_loss_fn = nn.MSELoss() if args.prediction_mode == "regression" else nn.CrossEntropyLoss()

        def eval_one_end_to_end(name, loader, encoder, quantizer, metadata) -> Dict[str, Any]:
            y_true_all, y_pred_all = [], []
            total_loss = 0.0

            with torch.no_grad():
                for batch in tqdm(loader, desc=f"[eval_risk:{name}]"):
                    batch = quantizer.transform_inplace(batch).to(device)

                    z, _, _ = encoder(batch)
                    g = batched_graph_embeddings(z, batch, metadata, embed_dim_per_type=args.embed_dim)
                    p = proj_head(g)
                    pred = risk_head(p)

                    if args.prediction_mode == "regression":
                        target = batch.y.view(-1, 1).float()
                        loss = risk_loss_fn(pred, target)
                        y_true_all.append(target.cpu().numpy())
                        y_pred_all.append(pred.cpu().numpy())
                    else:
                        target = risk_to_class_safe(batch.y).view(-1).long()
                        loss = risk_loss_fn(pred, target)
                        pred_cls = pred.argmax(dim=-1)
                        y_true_all.append(target.cpu().numpy())
                        y_pred_all.append(pred_cls.cpu().numpy())

                    total_loss += float(loss.item())

            avg_loss = total_loss / max(len(loader), 1)

            y_true_np = np.concatenate(y_true_all)
            y_pred_np = np.concatenate(y_pred_all)

            if args.prediction_mode == "regression":
                y_true_np = y_true_np.reshape(-1)
                y_pred_np = y_pred_np.reshape(-1)

            metrics: Dict[str, Any] = {f"eval/{name}_loss": avg_loss}

            print(f"\n========== {name} evaluation results ==========")
            if args.prediction_mode == "classification":
                cm = confusion_matrix(y_true_np, y_pred_np, labels=range(args.num_classes))
                cls_metrics = classification_metrics_from_cm(cm)
                cls_metrics["ordinal_mae_bins"] = mean_absolute_error(y_true_np, y_pred_np)
                cls_metrics["qwk"] = cohen_kappa_score(y_true_np, y_pred_np, weights="quadratic")

                metrics.update({f"eval/{name}_{k}": v for k, v in cls_metrics.items()})
                cm_list = metrics.pop(f"eval/{name}_confusion_matrix")

                for k, v in sorted(metrics.items()):
                    if isinstance(v, list):
                        print(f"{k}: {v}")
                    else:
                        print(f"{k}: {v:.4f}")

                print("\nConfusion matrix (rows=true, cols=pred):")
                print(_format_confusion_matrix(np.array(cm_list)))

            else:
                metrics[f"eval/{name}_mae"] = mean_absolute_error(y_true_np, y_pred_np)
                metrics[f"eval/{name}_rmse"] = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
                metrics[f"eval/{name}_r2"] = r2_score(y_true_np, y_pred_np)
                rho, p_val = spearmanr(y_true_np, y_pred_np)
                metrics[f"eval/{name}_spearman_rho"] = rho
                metrics[f"eval/{name}_spearman_p"] = p_val
                for k, v in sorted(metrics.items()):
                    print(f"{k}: {v:.4f}")

            print("===============================================\n")
            wb_log(metrics, bump=True)

            return {k.replace(f"eval/{name}_", ""): v for k, v in metrics.items()}

        nup_res = eval_one_end_to_end("nuplan_eval", nup_eval_loader, nup_enc, nup_quant, nup_eval_full.get_metadata())
        l2d_res = eval_one_end_to_end("l2d_eval", l2d_eval_loader, l2d_enc, l2d_quant, l2d_eval_full.get_metadata())

        eval_results_path = os.path.join(run_dir, "evaluation_results.json")
        results: Dict[str, Any] = {}
        if os.path.exists(eval_results_path):
            try:
                with open(eval_results_path, "r") as f:
                    results = json.load(f)
            except Exception:
                results = {}

        results.setdefault("4Ca_joint", {})
        results["4Ca_joint"]["nuplan_eval"] = nup_res
        results["4Ca_joint"]["l2d_eval"] = l2d_res
        results["4Ca_joint"]["proj_risk_ckpt_path"] = os.path.abspath(args.proj_risk_ckpt_path)
        results["4Ca_joint"]["l2d_encoder_ckpt_path"] = os.path.abspath(args.l2d_4b_ckpt_path)
        results["4Ca_joint"]["nuplan_encoder_ckpt_path"] = os.path.abspath(args.nup_4b_ckpt_path)

        with open(eval_results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"[ok] saved evaluation results -> {eval_results_path}")

    if wandb_run is not None:
        wandb_run.finish()


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4Ca: recon AE(s) then joint proj+align+risk (+distill) training.")

    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--output_root", type=str, default="./outputs")
    parser.add_argument("--run_id", type=str, default=None)

    parser.add_argument("--train_encoders_recon", type=_str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--train_proj_risk_jointly", type=_str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--evaluate_risk", type=_str2bool, nargs="?", const=True, default=False)

    parser.add_argument("--mode", type=str, default="all")
    parser.add_argument("--side_info_path", type=str, default=None)
    parser.add_argument("--unified_tags_path", type=str, default="data/training_data/unified_tags.json")
    parser.add_argument("--node_features_to_exclude", nargs="+", type=str, default=None)

    parser.add_argument("--quant_bins", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--num_encoder_layers", type=int, default=1)
    parser.add_argument("--num_decoder_layers", type=int, default=1)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--dropout_rate", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=10)

    parser.add_argument("--proj_risk_lr", type=float, default=1e-4)
    parser.add_argument("--proj_risk_weight_decay", type=float, default=1e-6)
    parser.add_argument("--proj_risk_epochs", type=int, default=10)

    parser.add_argument("--align_weight", type=float, default=0.1)
    parser.add_argument("--align_warmup_epochs", type=int, default=0)
    parser.add_argument("--align_ramp_epochs", type=int, default=0)

    parser.add_argument("--risk_loss_weight", type=float, default=1.0)

    parser.add_argument("--distill_weight", type=float, default=0.1)
    parser.add_argument("--distill_warmup_epochs", type=int, default=0)
    parser.add_argument("--distill_ramp_epochs", type=int, default=0)
    parser.add_argument("--distill_temp", type=float, default=2.0)
    parser.add_argument("--tag_softmax_temp", type=float, default=0.3)

    parser.add_argument("--prediction_mode", type=str, default="classification", choices=["regression", "classification"])
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--risk_hidden_dim", type=int, default=64)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=_str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--val_fraction", type=float, default=0.1)

    parser.add_argument("--encoder_ckpt_path", type=str, default="./models/graph_encoder/best_model.pt")
    parser.add_argument("--proj_risk_ckpt_path", type=str, default="./models/risk_from_embeddings/best_model.pt")

    parser.add_argument("--load_pretrained_4b_encoders", type=_str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--l2d_4b_ckpt_path", type=str, default=None)
    parser.add_argument("--nup_4b_ckpt_path", type=str, default=None)
    parser.add_argument("--force_retrain_encoders", type=_str2bool, nargs="?", const=True, default=False)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_wandb", type=_str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args()
    args = apply_yaml_overrides(parser, args)
    run(args)
