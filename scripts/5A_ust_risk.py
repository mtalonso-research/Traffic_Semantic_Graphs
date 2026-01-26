import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import random_split, ConcatDataset, Subset
from torch_geometric.loader import DataLoader
from scipy.stats import spearmanr
import wandb
from sklearn.metrics import (
    mean_absolute_error,
    cohen_kappa_score,
    mean_squared_error,
    r2_score,
    confusion_matrix,
)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.graph_encoding.data_loaders import get_graph_dataset
from src.graph_encoding.autoencoder import (
    batched_graph_embeddings,
    QuantileFeatureQuantizer,
    feature_loss,
    edge_loss,
    HeteroGraphAutoencoder,
)
from src.graph_encoding.risk_prediction import RiskPredictionHead
from src.experiment_utils import (
    set_seed,
    seed_worker,
    risk_to_class_safe,
    infer_graph_emb_dim,
    apply_yaml_overrides,
    resolve_paths,
    _format_confusion_matrix,
    _print_access,
    _require_dir,
    _require_file,
    _make_fallback_run_id,
    _ensure_dir,
    classification_metrics_from_cm,
    ProjectionHead,
    PairedAnchorDataset,
    _paired_alignment_loss,
    _paired_consistency_loss,
    _adopt_args_from_ckpt,
    _set_dataset_flags,
    _canonical_dataset_name,
    _set_requires_grad,
    log_annotations,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_AE_ARGS_TO_ADOPT = [
    "mode",
    "hidden_dim",
    "embed_dim",
    "num_encoder_layers",
    "num_decoder_layers",
    "activation",
    "dropout_rate",
    "quant_bins",
]

_STAGE2_ARGS_TO_ADOPT = [
    "prediction_mode",
    "num_classes",
    "risk_hidden_dim",
    "proj_hidden_dim",
    "proj_dropout",
    "proj_activation",
    "proj_residual",
    "proj_l2_normalize",
    "use_proj_clean",
    "align_weight",
    "align_loss_kind",
    "consistency_weight",
    "consistency_kind",
    "dataset_clean",
    "dataset_noisy",
]

def run_task(args: argparse.Namespace):
    # ---------------- Eval-only compatibility load ----------------
    is_eval_only_flag = bool(args.evaluate) and not bool(args.train_autoencoders) and not bool(args.train_stage2)
    if is_eval_only_flag:
        original_main_ckpt = args.best_model_path
        original_wandb = args.wandb
        original_clean = args.clean
        original_noisy = args.noisy
        _require_file(original_main_ckpt, "Main checkpoint (--best_model_path) for evaluation")
        ckpt = torch.load(original_main_ckpt, map_location="cpu", weights_only=False)
        if "args" in ckpt:
            print("[info] Overwriting CLI args with args from MAIN checkpoint for compatibility.")
            saved_args = ckpt["args"]
            for k, v in saved_args.items():
                if hasattr(args, k):
                    setattr(args, k, v)
            args.evaluate = True
            args.train_autoencoders = False
            args.train_stage2 = False
            args.best_model_path = original_main_ckpt
            args.wandb = original_wandb
            args.clean = original_clean
            args.noisy = original_noisy
        else:
            print("[warn] Main checkpoint does not contain 'args'. Using CLI args may cause mismatch.")

    # ---------------- W&B init ----------------
    wandb_run = None
    run_id = None
    if args.wandb or args.sweep:
        wandb_run = wandb.init(config=vars(args))
        run_id = wandb_run.id
        if args.sweep:
            for k, v in wandb.config.items():
                if hasattr(args, k):
                    setattr(args, k, v)

    if run_id is None:
        run_id = _make_fallback_run_id()
    if getattr(args, "run_id", None):
        run_id = args.run_id

    # ---------------- seed / paths ----------------
    set_seed(args.seed)
    if args.l2d:
        args.data_root = os.path.join("data", "L2D")
    elif args.nup:
        args.data_root = os.path.join("data", "NuPlan")
    args.data_root = os.path.abspath(args.data_root)

    ds_clean_name = "clean"
    ds_noisy_name = f"noisy_{args.noisy}"
    ds_noisy_true_name = "noisy_true"

    paths_clean_base = resolve_paths(args, ds_clean_name)
    paths_noisy = resolve_paths(args, ds_noisy_name)
    paths_noisy_true = resolve_paths(args, ds_noisy_true_name)

    # ---------------- verify train inputs ----------------
    _print_access("CLEAN BASE TRAIN graphs root", paths_clean_base["train_graph_root"])
    _print_access("CLEAN BASE TRAIN risk scores", paths_clean_base["train_risk_path"])
    _require_dir(paths_clean_base["train_graph_root"], "Clean base training graphs")
    _require_file(paths_clean_base["train_risk_path"], "Clean base training risk scores JSON")

    _print_access("NOISY TRAIN graphs root", paths_noisy["train_graph_root"])
    _print_access("NOISY TRAIN risk scores", paths_noisy["train_risk_path"])
    _require_dir(paths_noisy["train_graph_root"], "Noisy training graphs")
    _require_file(paths_noisy["train_risk_path"], "Noisy training risk scores JSON")

    _print_access("NOISY TRUE TRAIN graphs root", paths_noisy_true["train_graph_root"])
    _print_access("NOISY TRUE TRAIN risk scores", paths_noisy_true["train_risk_path"])
    _require_dir(paths_noisy_true["train_graph_root"], "Noisy true training graphs")
    _require_file(paths_noisy_true["train_risk_path"], "Noisy true training risk scores JSON")

    # ---------------- load TRAIN datasets ----------------
    print(f"Loading TRAIN datasets: clean_base, noisy={ds_noisy_name}, noisy_true={ds_noisy_true_name}...")
    base_clean_dataset_full = get_graph_dataset(
        root_dir=paths_clean_base["train_graph_root"],
        mode=args.mode,
        side_information_path=None,
        risk_scores_path=paths_clean_base["train_risk_path"],
    )
    noisy_dataset_full = get_graph_dataset(
        root_dir=paths_noisy["train_graph_root"],
        mode=args.mode,
        side_information_path=None,
        risk_scores_path=paths_noisy["train_risk_path"],
    )
    noisy_true_dataset_full = get_graph_dataset(
        root_dir=paths_noisy_true["train_graph_root"],
        mode=args.mode,
        side_information_path=None,
        risk_scores_path=paths_noisy_true["train_risk_path"],
    )

    # ---------------- Create final clean dataset with anchors ----------------
    meta_clean = base_clean_dataset_full.get_metadata()
    meta_noisy = noisy_dataset_full.get_metadata()
    num_base_clean = len(base_clean_dataset_full)
    num_anchors = int((args.clean / 100) * num_base_clean)
    print('NUM ANCHORS', num_anchors)
    num_to_keep_from_clean = num_base_clean - num_anchors

    if num_anchors > 0:
        # Get indices to keep from base clean dataset
        clean_indices = np.random.permutation(num_base_clean)
        keep_clean_indices = clean_indices[:num_to_keep_from_clean]
        
        # Get indices for anchors from noisy_true dataset
        noisy_true_indices = np.random.permutation(len(noisy_true_dataset_full))
        anchor_indices = noisy_true_indices[:num_anchors]

        # Create subset of base clean dataset
        clean_subset = Subset(base_clean_dataset_full, keep_clean_indices)
        
        # Create subset of noisy_true dataset for anchors
        anchor_subset = Subset(noisy_true_dataset_full, anchor_indices)

        # Combine to create the final clean dataset
        clean_dataset_full = ConcatDataset([clean_subset, anchor_subset])
        print(f"[info] Created final clean dataset with {len(clean_subset)} clean samples and {len(anchor_subset)} anchors.")
    else:
        clean_dataset_full = base_clean_dataset_full
        print("[info] Using base clean dataset without anchors.")
    
    print(f"[info] Total samples in final clean dataset: {len(clean_dataset_full)}")



    # ---------------- output isolation ----------------
    is_eval_only = bool(args.evaluate) and not bool(args.train_autoencoders) and not bool(args.train_stage2)
    if is_eval_only:
        main_ckpt_path = os.path.abspath(args.best_model_path)
        _require_file(main_ckpt_path, "Main checkpoint (--best_model_path) for evaluation")
        run_dir = os.path.dirname(main_ckpt_path)
        eval_results_path = os.path.join(run_dir, "evaluation_results.json")
        if wandb_run is not None:
            wandb.config.update({"run_id": run_id, "run_dir": run_dir, "best_model_path": main_ckpt_path}, allow_val_change=True)
    else:
        output_root = _ensure_dir(os.path.abspath(args.output_root))
        run_dir = _ensure_dir(
            os.path.join(
                output_root,
                "4Bb_align_anchors_paired_FIXED",
                f"{ds_clean_name}_vs_{ds_noisy_name}",
                ("classification" if args.prediction_mode == "classification" else "regression"),
                run_id,
            )
        )
        if wandb_run is not None:
            wandb.config.update({"run_id": run_id, "run_dir": run_dir}, allow_val_change=True)

        main_ckpt_name = os.path.basename(args.best_model_path)
        if main_ckpt_name == "best_model.pt":
            main_ckpt_name = f"4Bb_{ds_clean_name}_vs_{ds_noisy_name}_anchors_paired_FIXED_best_model.pt"
        main_ckpt_path = os.path.join(run_dir, main_ckpt_name)
        eval_results_path = os.path.join(run_dir, "evaluation_results.json")

    ae_clean_ckpt_path = os.path.join(run_dir, f"4Bb_{ds_clean_name}_ae_best_model.pt")
    ae_noisy_ckpt_path = os.path.join(run_dir, f"4Bb_{ds_noisy_name}_ae_best_model.pt")

    # ---------------- fit quantizers per domain ----------------
    print("Fitting quantizers on TRAIN datasets...")
    quant_clean = QuantileFeatureQuantizer(bins=args.quant_bins, node_types=meta_clean[0])
    quant_clean.fit(clean_dataset_full)
    quant_noisy = QuantileFeatureQuantizer(bins=args.quant_bins, node_types=meta_noisy[0])
    quant_noisy.fit(noisy_dataset_full)

    # ---------------- deterministic split per domain ----------------
    def split_dataset(ds, seed: int, val_fraction: float):
        val_size = int(val_fraction * len(ds))
        train_size = len(ds) - val_size
        gen = torch.Generator().manual_seed(seed)
        return random_split(ds, [train_size, val_size], generator=gen)

    clean_train, clean_val = split_dataset(clean_dataset_full, args.seed, args.val_fraction)
    noisy_train, noisy_val = split_dataset(noisy_dataset_full, args.seed, args.val_fraction)

    loader_gen = torch.Generator().manual_seed(args.seed)

    common_loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=loader_gen,
    )
    clean_train_loader = DataLoader(clean_train, shuffle=True, **common_loader_kwargs)
    clean_val_loader = DataLoader(clean_val, shuffle=False, **common_loader_kwargs)
    noisy_train_loader = DataLoader(noisy_train, shuffle=True, **common_loader_kwargs)
    noisy_val_loader = DataLoader(noisy_val, shuffle=False, **common_loader_kwargs)

    paired_train = PairedAnchorDataset(clean_train, noisy_train)
    paired_val = PairedAnchorDataset(clean_val, noisy_val)

    paired_train_loader = DataLoader(
        paired_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=loader_gen,
    )
    paired_val_loader = DataLoader(
        paired_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=loader_gen,
    )

    print(f"[info] paired anchors: train={len(paired_train)} val={len(paired_val)}")

    # ---------------- build AEs ----------------
    print("Initializing autoencoders...")

    encoder_clean = HeteroGraphAutoencoder(
        metadata=meta_clean,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        quantizer_spec=quant_clean.spec(),
        feat_emb_dim=16,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        activation=args.activation,
        dropout_rate=args.dropout_rate,
        side_info_dim=0,
    ).to(device)

    encoder_noisy = HeteroGraphAutoencoder(
        metadata=meta_noisy,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        quantizer_spec=quant_noisy.spec(),
        feat_emb_dim=16,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        activation=args.activation,
        dropout_rate=args.dropout_rate,
        side_info_dim=0,
    ).to(device)

    graph_emb_dim_clean = infer_graph_emb_dim(
        encoder_clean, quant_clean, clean_train_loader, meta_clean, embed_dim_per_type=args.embed_dim
    )
    graph_emb_dim_noisy = infer_graph_emb_dim(
        encoder_noisy, quant_noisy, noisy_train_loader, meta_noisy, embed_dim_per_type=args.embed_dim
    )

    print(f"[info] inferred graph_emb_dim_clean = {graph_emb_dim_clean}")
    print(f"[info] inferred graph_emb_dim_noisy = {graph_emb_dim_noisy}")
    if graph_emb_dim_clean != graph_emb_dim_noisy:
        raise SystemExit(
            f"Graph embedding dims differ: clean={graph_emb_dim_clean}, noisy={graph_emb_dim_noisy}. "
            "Projection heads are same-dim by design."
        )
    graph_emb_dim = graph_emb_dim_clean

    # ---------------- stage helpers ----------------
    def encode_graph_embeddings_clean(batch):
        batch = quant_clean.transform_inplace(batch).to(device)
        z_dict, feat_logits, edge_logits = encoder_clean(batch)
        g = batched_graph_embeddings(z_dict, batch, meta_clean, embed_dim_per_type=args.embed_dim)
        return batch, z_dict, feat_logits, edge_logits, g

    def encode_graph_embeddings_noisy(batch):
        batch = quant_noisy.transform_inplace(batch).to(device)
        z_dict, feat_logits, edge_logits = encoder_noisy(batch)
        g = batched_graph_embeddings(z_dict, batch, meta_noisy, embed_dim_per_type=args.embed_dim)
        return batch, z_dict, feat_logits, edge_logits, g

    # ---------------- optionally load AE checkpoints (adopt args) ----------------
    if args.load_best_ae_clean:
        _require_file(args.ae_clean_ckpt_path, "--ae_clean_ckpt_path for --load_best_ae_clean")
        ck = torch.load(args.ae_clean_ckpt_path, map_location="cpu", weights_only=False)
        _adopt_args_from_ckpt(args, ck, _AE_ARGS_TO_ADOPT, label="CLEAN AE")
        encoder_clean.load_state_dict(ck["encoder_state_dict"])
        encoder_clean.eval()
        print(f"[ok] loaded CLEAN AE checkpoint: {args.ae_clean_ckpt_path}")

    if args.load_best_ae_noisy:
        _require_file(args.ae_noisy_ckpt_path, "--ae_noisy_ckpt_path for --load_best_ae_noisy")
        ck = torch.load(args.ae_noisy_ckpt_path, map_location="cpu", weights_only=False)
        _adopt_args_from_ckpt(args, ck, _AE_ARGS_TO_ADOPT, label="NOISY AE")
        encoder_noisy.load_state_dict(ck["encoder_state_dict"])
        encoder_noisy.eval()
        print(f"[ok] loaded NOISY AE checkpoint: {args.ae_noisy_ckpt_path}")

    # ---------------- Stage 1: train AEs (optional) ----------------
    if args.train_autoencoders:
        print(f"Stage 1/2: Training AUTOENCODERS for clean={ds_clean_name} and noisy={ds_noisy_name}...")

        def train_one_ae(encoder, encode_fn, train_loader, val_loader, ckpt_path: str, label: str):
            best_val = float("inf")
            opt = torch.optim.Adam(encoder.parameters(), lr=args.ae_lr, weight_decay=args.ae_weight_decay)

            for epoch in range(1, args.ae_epochs + 1):
                encoder.train()
                total = 0.0
                nb = 0
                for batch in tqdm(train_loader, desc=f"AE[{label}] Epoch {epoch:02d} [train]"):
                    opt.zero_grad()
                    batch, z, feat_logits, edge_logits, _g = encode_fn(batch)
                    l_feat = feature_loss(feat_logits, batch)
                    l_edge = edge_loss(edge_logits, z, encoder.edge_decoders)
                    loss = l_feat + l_edge
                    loss.backward()
                    opt.step()
                    total += float(loss.item())
                    nb += 1
                train_loss = total / max(nb, 1)

                encoder.eval()
                vtotal = 0.0
                vn = 0
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc=f"AE[{label}] Epoch {epoch:02d} [val]"):
                        batch, z, feat_logits, edge_logits, _g = encode_fn(batch)
                        l_feat = feature_loss(feat_logits, batch)
                        l_edge = edge_loss(edge_logits, z, encoder.edge_decoders)
                        loss = l_feat + l_edge
                        vtotal += float(loss.item())
                        vn += 1

                val_loss = vtotal / max(vn, 1)
                print(f"AE[{label}] Epoch {epoch:02d} | train_recon={train_loss:.4f} | val_recon={val_loss:.4f}")

                if wandb_run is not None:
                    wandb.log(
                        {
                            f"ae_{label}/epoch": epoch,
                            f"ae_{label}/train_recon": train_loss,
                            f"ae_{label}/val_recon": val_loss,
                        }
                    )

                if val_loss < best_val:
                    best_val = val_loss
                    _ensure_dir(os.path.dirname(ckpt_path))
                    torch.save(
                        {
                            "stage": f"autoencoder_best_{label}",
                            "encoder_state_dict": encoder.state_dict(),
                            "best_ae_val_recon": best_val,
                            "graph_emb_dim": graph_emb_dim,
                            "dataset_label": label,
                            "args": dict(vars(args)),
                            "run_id": run_id,
                            "run_dir": run_dir,
                        },
                        ckpt_path,
                    )
                    print(f"  -> New best AE[{label}] saved to {ckpt_path}")

        train_one_ae(encoder_clean, encode_graph_embeddings_clean, clean_train_loader, clean_val_loader, ae_clean_ckpt_path, label="clean")
        train_one_ae(encoder_noisy, encode_graph_embeddings_noisy, noisy_train_loader, noisy_val_loader, ae_noisy_ckpt_path, label="noisy")

    # ---------------- Freeze encoders for stage 2 ----------------
    _set_requires_grad(encoder_clean, False)
    _set_requires_grad(encoder_noisy, False)
    encoder_clean.eval()
    encoder_noisy.eval()

    # ---------------- Stage 2 models ----------------
    proj_clean = None
    if args.use_proj_clean:
        proj_clean = ProjectionHead(
            dim=graph_emb_dim,
            hidden_dim=args.proj_hidden_dim if args.proj_hidden_dim > 0 else None,
            dropout=args.proj_dropout,
            activation=args.proj_activation,
            residual=bool(args.proj_residual),
        ).to(device)

    proj_noisy = ProjectionHead(
        dim=graph_emb_dim,
        hidden_dim=args.proj_hidden_dim if args.proj_hidden_dim > 0 else None,
        dropout=args.proj_dropout,
        activation=args.proj_activation,
        residual=bool(args.proj_residual),
    ).to(device)

    out_dim = 1 if args.prediction_mode == "regression" else args.num_classes
    risk_head = RiskPredictionHead(
        input_dim=graph_emb_dim,
        hidden_dim=args.risk_hidden_dim,
        output_dim=out_dim,
        mode=args.prediction_mode,
    ).to(device)

    risk_loss_fn = nn.MSELoss() if args.prediction_mode == "regression" else nn.CrossEntropyLoss()

    # ---------------- load pretrained stage2 parts (optional) ----------------
    if args.load_risk_head:
        _require_file(args.risk_head_ckpt_path, "--risk_head_ckpt_path for --load_risk_head")
        ck = torch.load(args.risk_head_ckpt_path, map_location=device, weights_only=False)
        if "risk_head_state_dict" not in ck:
            raise SystemExit("Risk-head checkpoint missing key: risk_head_state_dict")
        risk_head.load_state_dict(ck["risk_head_state_dict"])
        risk_head.eval()
        print(f"[ok] loaded pretrained risk_head from: {args.risk_head_ckpt_path}")

        if args.use_proj_clean and ("proj_clean_state_dict" in ck):
            proj_clean.load_state_dict(ck["proj_clean_state_dict"])
            proj_clean.eval()
            print("[ok] also loaded proj_clean from risk-head checkpoint.")

    if args.load_proj_noisy:
        _require_file(args.proj_noisy_ckpt_path, "--proj_noisy_ckpt_path for --load_proj_noisy")
        ck = torch.load(args.proj_noisy_ckpt_path, map_location=device, weights_only=False)
        if "proj_noisy_state_dict" not in ck:
            raise SystemExit("proj_noisy checkpoint missing key: proj_noisy_state_dict")
        proj_noisy.load_state_dict(ck["proj_noisy_state_dict"])
        proj_noisy.eval()
        print(f"[ok] loaded pretrained proj_noisy from: {args.proj_noisy_ckpt_path}")

    # ---------------- Stage 2: train (optional) ----------------
    best_val_risk = float("inf")

    if args.train_stage2:
        print("Stage 2: training with CLEAN as canonical; updating ONLY the intended modules per loop...")

        # Risk optimizer updates risk_head (+ proj_clean if enabled).
        risk_params = list(risk_head.parameters())
        if proj_clean is not None:
            risk_params += list(proj_clean.parameters())
        opt_risk = torch.optim.Adam(risk_params, lr=args.stage2_lr, weight_decay=args.stage2_weight_decay)

        # Align optimizer updates proj_noisy only.
        opt_align = torch.optim.Adam(list(proj_noisy.parameters()), lr=args.stage2_lr, weight_decay=args.stage2_weight_decay)

        if args.train_noisy_proj_only and (not args.load_risk_head):
            raise SystemExit("--train_noisy_proj_only requires --load_risk_head (provide a pretrained risk head).")

        for epoch in range(1, args.stage2_epochs + 1):
            # ---------- TRAIN ----------
            risk_head.train()
            if proj_clean is not None:
                proj_clean.train()
            proj_noisy.train()

            total_loss_sum = 0.0
            risk_loss_sum = 0.0
            align_loss_sum = 0.0
            cons_loss_sum = 0.0
            steps = 0

            correct = 0
            seen = 0
            anchor_pairs_seen = 0

            # (A) Supervised risk training on clean (skip if training only proj_noisy)
            if not args.train_noisy_proj_only:
                _set_requires_grad(risk_head, True)
                if proj_clean is not None:
                    _set_requires_grad(proj_clean, True)

                for batch_c in tqdm(clean_train_loader, desc=f"Stage2 Epoch {epoch:02d} [train:clean_risk]"):
                    opt_risk.zero_grad()

                    batch_c, _z, _fl, _el, g_c = encode_graph_embeddings_clean(batch_c)
                    p_c = proj_clean(g_c) if proj_clean is not None else g_c
                    if args.proj_l2_normalize:
                        p_c = F.normalize(p_c, dim=-1)

                    logits_c = risk_head(p_c)
                    if args.prediction_mode == "regression":
                        target = batch_c.y.view(-1, 1).float()
                        l_risk = risk_loss_fn(logits_c, target)
                    else:
                        target = risk_to_class_safe(batch_c.y)
                        l_risk = risk_loss_fn(logits_c, target)
                        pred_cls = torch.argmax(logits_c, dim=-1)
                        correct += int((pred_cls == target).sum().item())
                        seen += int(target.numel())

                    l_risk.backward()
                    opt_risk.step()

                    total_loss_sum += float(l_risk.item())
                    risk_loss_sum += float(l_risk.item())
                    steps += 1

            # (B) Anchor alignment (update proj_noisy ONLY; freeze risk_head and proj_clean)
            _set_requires_grad(risk_head, False)
            risk_head.eval()
            if proj_clean is not None:
                _set_requires_grad(proj_clean, False)
                proj_clean.eval()

            for batch_c, batch_n in tqdm(paired_train_loader, desc=f"Stage2 Epoch {epoch:02d} [train:anchors]"):
                opt_align.zero_grad()

                batch_c, _z, _fl, _el, g_c = encode_graph_embeddings_clean(batch_c)
                batch_n, _z2, _fl2, _el2, g_n = encode_graph_embeddings_noisy(batch_n)

                # Canonical target embedding from clean
                t_c = proj_clean(g_c) if proj_clean is not None else g_c
                s_n = proj_noisy(g_n)

                if args.proj_l2_normalize:
                    t_c = F.normalize(t_c, dim=-1)
                    s_n = F.normalize(s_n, dim=-1)

                l_align = _paired_alignment_loss(t_c, s_n, loss_kind=args.align_loss_kind)
                anchor_pairs_seen += int(t_c.shape[0])

                l_cons = torch.tensor(0.0, device=device)
                if args.consistency_weight > 0.0:
                    # teacher: clean -> risk_head (no grad); student: noisy mapped -> risk_head (no grad to head)
                    with torch.no_grad():
                        logits_teacher = risk_head(t_c)
                    logits_student = risk_head(s_n)  # grads flow into s_n -> proj_noisy only (risk_head frozen)
                    l_cons = _paired_consistency_loss(
                        logits_teacher=logits_teacher,
                        logits_student=logits_student,
                        mode=args.prediction_mode,
                        kind=args.consistency_kind,
                    )

                loss = (args.align_weight * l_align) + (args.consistency_weight * l_cons)
                loss.backward()
                opt_align.step()

                total_loss_sum += float(loss.item())
                align_loss_sum += float(l_align.item())
                cons_loss_sum += float(l_cons.item()) if args.consistency_weight > 0.0 else 0.0
                steps += 1

            train_total = total_loss_sum / max(steps, 1)
            train_risk = risk_loss_sum / max(steps, 1)
            train_align = align_loss_sum / max(steps, 1)
            train_cons = cons_loss_sum / max(steps, 1)
            train_acc = (correct / max(seen, 1)) if (args.prediction_mode == "classification" and not args.train_noisy_proj_only) else None

            # ---------- VAL ----------
            risk_head.eval()
            if proj_clean is not None:
                proj_clean.eval()
            proj_noisy.eval()

            # (A) val risk on clean
            v_risk_sum = 0.0
            v_risk_steps = 0
            v_correct = 0
            v_seen = 0

            with torch.no_grad():
                for batch_c in tqdm(clean_val_loader, desc=f"Stage2 Epoch {epoch:02d} [val:clean_risk]"):
                    batch_c, _z, _fl, _el, g_c = encode_graph_embeddings_clean(batch_c)
                    p_c = proj_clean(g_c) if proj_clean is not None else g_c
                    if args.proj_l2_normalize:
                        p_c = F.normalize(p_c, dim=-1)

                    logits_c = risk_head(p_c)
                    if args.prediction_mode == "regression":
                        target = batch_c.y.view(-1, 1).float()
                        l_risk = risk_loss_fn(logits_c, target)
                    else:
                        target = risk_to_class_safe(batch_c.y)
                        l_risk = risk_loss_fn(logits_c, target)
                        pred_cls = torch.argmax(logits_c, dim=-1)
                        v_correct += int((pred_cls == target).sum().item())
                        v_seen += int(target.numel())

                    v_risk_sum += float(l_risk.item())
                    v_risk_steps += 1

            val_risk = v_risk_sum / max(v_risk_steps, 1)
            val_acc = (v_correct / max(v_seen, 1)) if args.prediction_mode == "classification" else None

            # (B) val align/cons on paired anchors (logging only)
            v_align_sum = 0.0
            v_cons_sum = 0.0
            v_steps2 = 0
            v_pairs = 0

            with torch.no_grad():
                for batch_c, batch_n in tqdm(paired_val_loader, desc=f"Stage2 Epoch {epoch:02d} [val:anchors]"):
                    batch_c, _z, _fl, _el, g_c = encode_graph_embeddings_clean(batch_c)
                    batch_n, _z2, _fl2, _el2, g_n = encode_graph_embeddings_noisy(batch_n)

                    t_c = proj_clean(g_c) if proj_clean is not None else g_c
                    s_n = proj_noisy(g_n)

                    if args.proj_l2_normalize:
                        t_c = F.normalize(t_c, dim=-1)
                        s_n = F.normalize(s_n, dim=-1)

                    l_align = _paired_alignment_loss(t_c, s_n, loss_kind=args.align_loss_kind)

                    l_cons = 0.0
                    if args.consistency_weight > 0.0:
                        logits_teacher = risk_head(t_c)
                        logits_student = risk_head(s_n)
                        l_cons = float(_paired_consistency_loss(
                            logits_teacher=logits_teacher,
                            logits_student=logits_student,
                            mode=args.prediction_mode,
                            kind=args.consistency_kind,
                        ).item())

                    v_align_sum += float(l_align.item())
                    v_cons_sum += float(l_cons)
                    v_steps2 += 1
                    v_pairs += int(t_c.shape[0])

            val_align = v_align_sum / max(v_steps2, 1)
            val_cons = v_cons_sum / max(v_steps2, 1)

            val_total = val_risk + (args.align_weight * val_align) + (args.consistency_weight * val_cons)

            if args.prediction_mode == "classification":
                if train_acc is None:
                    print(
                        f"Stage2 Epoch {epoch:02d} | train_total={train_total:.4f} | val_total={val_total:.4f} | "
                        f"val_risk={val_risk:.4f} | val_acc={val_acc:.4f} | "
                        f"train_align={train_align:.4f} | val_align={val_align:.4f} | "
                        f"train_cons={train_cons:.4f} | val_cons={val_cons:.4f} | "
                        f"anchors(train/val)={anchor_pairs_seen}/{v_pairs}"
                    )
                else:
                    print(
                        f"Stage2 Epoch {epoch:02d} | train_total={train_total:.4f} | val_total={val_total:.4f} | "
                        f"train_risk={train_risk:.4f} | val_risk={val_risk:.4f} | "
                        f"train_align={train_align:.4f} | val_align={val_align:.4f} | "
                        f"train_cons={train_cons:.4f} | val_cons={val_cons:.4f} | "
                        f"anchors(train/val)={anchor_pairs_seen}/{v_pairs} | "
                        f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f}"
                    )
            else:
                print(
                    f"Stage2 Epoch {epoch:02d} | train_total={train_total:.4f} | val_total={val_total:.4f} | "
                    f"train_risk={train_risk:.4f} | val_risk={val_risk:.4f} | "
                    f"train_align={train_align:.4f} | val_align={val_align:.4f} | "
                    f"train_cons={train_cons:.4f} | val_cons={val_cons:.4f} | "
                    f"anchors(train/val)={anchor_pairs_seen}/{v_pairs}"
                )

            if wandb_run is not None:
                payload = {
                    "stage2/epoch": epoch,
                    "stage2/train_total": train_total,
                    "stage2/val_total": val_total,
                    "stage2/train_risk": train_risk,
                    "stage2/val_risk": val_risk,
                    "stage2/train_align": train_align,
                    "stage2/val_align": val_align,
                    "stage2/train_cons": train_cons,
                    "stage2/val_cons": val_cons,
                    "stage2/anchors_train_pairs": float(anchor_pairs_seen),
                    "stage2/anchors_val_pairs": float(v_pairs),
                }
                if args.prediction_mode == "classification":
                    if train_acc is not None:
                        payload.update({"stage2/train_acc": train_acc})
                    payload.update({"stage2/val_acc": val_acc})
                wandb.log(payload)

            if val_risk < best_val_risk:
                best_val_risk = val_risk
                _ensure_dir(os.path.dirname(main_ckpt_path))
                ckpt_out = {
                    "stage": "align_proj_noisy_risk_best_FIXED",
                    "encoder_clean_state_dict": encoder_clean.state_dict(),
                    "encoder_noisy_state_dict": encoder_noisy.state_dict(),
                    "proj_noisy_state_dict": proj_noisy.state_dict(),
                    "risk_head_state_dict": risk_head.state_dict(),
                    "best_val_risk": best_val_risk,
                    "graph_emb_dim": graph_emb_dim,
                    "dataset_clean": ds_clean_name,
                    "dataset_noisy": ds_noisy_name,
                    "args": dict(vars(args)),
                    "run_id": run_id,
                    "run_dir": run_dir,
                    "paired_train_anchors": len(paired_train),
                    "paired_val_anchors": len(paired_val),
                    "use_proj_clean": bool(args.use_proj_clean),
                }
                if proj_clean is not None:
                    ckpt_out["proj_clean_state_dict"] = proj_clean.state_dict()

                torch.save(ckpt_out, main_ckpt_path)
                print(f"  -> New best FIXED model saved to {main_ckpt_path}")

    # ---------------- EVALUATE ----------------
    if args.evaluate:
        print(f"Evaluating FIXED paired-anchor model: clean={ds_clean_name}, noisy={ds_noisy_name}...")

        _require_file(main_ckpt_path, "Main checkpoint (--best_model_path or run output)")
        ckpt = torch.load(main_ckpt_path, map_location=device, weights_only=False)

        _adopt_args_from_ckpt(args, ckpt, _STAGE2_ARGS_TO_ADOPT, label="MAIN")

        encoder_clean.load_state_dict(ckpt["encoder_clean_state_dict"])
        encoder_noisy.load_state_dict(ckpt["encoder_noisy_state_dict"])
        proj_noisy.load_state_dict(ckpt["proj_noisy_state_dict"])
        risk_head.load_state_dict(ckpt["risk_head_state_dict"])

        encoder_clean.eval()
        encoder_noisy.eval()
        proj_noisy.eval()
        risk_head.eval()

        # optional proj_clean
        proj_clean_eval = None
        if bool(ckpt.get("use_proj_clean", False)):
            proj_clean_eval = ProjectionHead(
                dim=graph_emb_dim,
                hidden_dim=args.proj_hidden_dim if args.proj_hidden_dim > 0 else None,
                dropout=args.proj_dropout,
                activation=args.proj_activation,
                residual=bool(args.proj_residual),
            ).to(device)
            if "proj_clean_state_dict" not in ckpt:
                raise SystemExit("Checkpoint says use_proj_clean=True but proj_clean_state_dict missing.")
            proj_clean_eval.load_state_dict(ckpt["proj_clean_state_dict"])
            proj_clean_eval.eval()

        _print_access("CLEAN EVAL graphs root", paths_clean_base["eval_graph_root"])
        _print_access("CLEAN EVAL risk scores", paths_clean_base["eval_risk_path"])
        _require_dir(paths_clean_base["eval_graph_root"], "Clean evaluation graphs")
        _require_file(paths_clean_base["eval_risk_path"], "Clean evaluation risk scores JSON")

        _print_access("NOISY EVAL graphs root", paths_noisy["eval_graph_root"])
        _print_access("NOISY EVAL risk scores", paths_noisy["eval_risk_path"])
        _require_dir(paths_noisy["eval_graph_root"], "Noisy evaluation graphs")
        _require_file(paths_noisy["eval_risk_path"], "Noisy evaluation risk scores JSON")

        print("Loading EVAL datasets...")
        clean_eval_ds = get_graph_dataset(
            root_dir=paths_clean_base["eval_graph_root"],
            mode=args.mode,
            side_information_path=None,
            risk_scores_path=paths_clean_base["eval_risk_path"],
        )
        noisy_eval_ds = get_graph_dataset(
            root_dir=paths_noisy["eval_graph_root"],
            mode=args.mode,
            side_information_path=None,
            risk_scores_path=paths_noisy["eval_risk_path"],
        )

        clean_eval_loader = DataLoader(
            clean_eval_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker if args.num_workers > 0 else None,
        )
        noisy_eval_loader = DataLoader(
            noisy_eval_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker if args.num_workers > 0 else None,
        )

        def eval_one(loader, domain: str):
            y_true_all, y_pred_all = [], []
            total_loss = 0.0
            nb = 0

            with torch.no_grad():
                for batch in tqdm(loader, desc=f"[eval:{domain}]"):
                    if domain == "clean":
                        batch, _z, _fl, _el, g = encode_graph_embeddings_clean(batch)
                        p = proj_clean_eval(g) if proj_clean_eval is not None else g
                    else:
                        batch, _z, _fl, _el, g = encode_graph_embeddings_noisy(batch)
                        p = proj_noisy(g)

                    if args.proj_l2_normalize:
                        p = F.normalize(p, dim=-1)

                    logits = risk_head(p)

                    if args.prediction_mode == "regression":
                        target = batch.y.view(-1, 1).float()
                        loss = risk_loss_fn(logits, target)
                        y_true_all.append(target.cpu().numpy())
                        y_pred_all.append(logits.cpu().numpy())
                    else:
                        target = risk_to_class_safe(batch.y)
                        loss = risk_loss_fn(logits, target)
                        pred_cls = torch.argmax(logits, dim=-1)
                        y_true_all.append(target.cpu().numpy())
                        y_pred_all.append(pred_cls.cpu().numpy())

                    total_loss += float(loss.item())
                    nb += 1

            avg_loss = total_loss / max(nb, 1)
            y_true_np = np.concatenate(y_true_all)
            y_pred_np = np.concatenate(y_pred_all)

            metrics = {f"{domain}/loss": avg_loss}

            if args.prediction_mode == "classification":
                cm = confusion_matrix(y_true_np, y_pred_np, labels=range(args.num_classes))
                cls_metrics = classification_metrics_from_cm(cm)
                cls_metrics["ordinal_mae_bins"] = mean_absolute_error(y_true_np, y_pred_np)
                cls_metrics["qwk"] = cohen_kappa_score(y_true_np, y_pred_np, weights="quadratic")
                cm_list = cls_metrics.pop("confusion_matrix")
                metrics.update({f"{domain}/{k}": v for k, v in cls_metrics.items()})
                return metrics, cm_list
            else:
                metrics[f"{domain}/mae"] = mean_absolute_error(y_true_np, y_pred_np)
                metrics[f"{domain}/rmse"] = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
                metrics[f"{domain}/r2"] = r2_score(y_true_np, y_pred_np)
                rho, pval = spearmanr(y_true_np, y_pred_np)
                metrics[f"{domain}/spearman_rho"] = rho
                metrics[f"{domain}/spearman_p"] = pval
                return metrics, None

        clean_metrics, clean_cm = eval_one(clean_eval_loader, "clean")
        noisy_metrics, noisy_cm = eval_one(noisy_eval_loader, "noisy")

        metrics = {}
        metrics.update({f"eval/{k}": v for k, v in clean_metrics.items()})
        metrics.update({f"eval/{k}": v for k, v in noisy_metrics.items()})

        print("\n========== 4Bb FIXED evaluation results ==========")
        for k, v in sorted(metrics.items()):
            if isinstance(v, list):
                print(f"{k}: {v}")
            else:
                try:
                    print(f"{k}: {float(v):.4f}")
                except Exception:
                    print(f"{k}: {v}")
        if args.prediction_mode == "classification":
            print("\nCLEAN confusion matrix (rows=true, cols=pred):")
            print(_format_confusion_matrix(np.array(clean_cm)))
            print("\nNOISY confusion matrix (rows=true, cols=pred):")
            print(_format_confusion_matrix(np.array(noisy_cm)))
        print("===============================================\n")

        if wandb_run is not None:
            wandb.log(metrics)

        if args.save_annotations:
            results = {}
            if os.path.exists(eval_results_path):
                import json
                with open(eval_results_path, "r") as f:
                    results = json.load(f)

            results.setdefault("4Bb_align_anchors_paired_FIXED", {})
            key = f"{ds_clean_name}_vs_{ds_noisy_name}"
            results["4Bb_align_anchors_paired_FIXED"][key + "_run_id"] = run_id
            results["4Bb_align_anchors_paired_FIXED"][key + "_run_dir"] = run_dir
            for k, v in metrics.items():
                results["4Bb_align_anchors_paired_FIXED"][key + "_" + k.replace("eval/", "")] = v

            import json
            with open(eval_results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved evaluation results to {eval_results_path}")

    if wandb_run is not None:
        wandb_run.finish()

    log_annotations(
        file_path="./experiment_results/df.csv",
        script_name="UST",
        anchor_pct=args.clean,
        noise_pct=args.noisy,
        seed=args.seed,
        metrics=metrics,
        domain=None, # 5A produces both clean and noisy metrics
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="4Bb (anchors, paired loader) FIXED: clean is canonical; train risk separately; train proj_noisy only on anchors."
    )

    # Dataset source
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--l2d", action="store_true", help="Use L2D dataset.")
    source_group.add_argument("--nup", action="store_true", help="Use NuPlan dataset.")
    parser.add_argument("--data_root", type=str, help="Root directory of the dataset.")

    # Dataset type
    parser.add_argument("--clean", type=int, required=True, help="Percentage of anchor samples from the noisy_true dataset.")
    parser.add_argument("--noisy", type=int, required=True, help="Percentage of noise in the noisy dataset (e.g., 10, 20).")


    # Model (AEs)
    parser.add_argument("--mode", type=str, default="all")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--num_encoder_layers", type=int, default=1)
    parser.add_argument("--num_decoder_layers", type=int, default=1)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--quant_bins", type=int, default=32)

    # Risk head
    parser.add_argument("--risk_hidden_dim", type=int, default=64)
    parser.add_argument("--prediction_mode", type=str, default="classification", choices=["regression", "classification"])
    parser.add_argument("--num_classes", type=int, default=4)

    # Projection heads
    parser.add_argument("--use_proj_clean", action="store_true", help="If set, learn a proj_clean; otherwise clean embeddings are canonical.")
    parser.add_argument("--proj_hidden_dim", type=int, default=0, help="0 => same dim as embedding")
    parser.add_argument("--proj_dropout", type=float, default=0.1)
    parser.add_argument("--proj_activation", type=str, default="relu", choices=["relu", "gelu"])
    parser.add_argument("--proj_residual", action="store_true")

    # L2 normalize projected embeddings before risk/alignment
    parser.add_argument("--proj_l2_normalize", action="store_true", help="L2-normalize embeddings before risk/alignment.")

    # Alignment by anchors
    parser.add_argument("--align_weight", type=float, default=1.0)
    parser.add_argument("--align_loss_kind", type=str, default="l2", choices=["l2", "smoothl1", "cosine"])

    # Consistency regularizer (per-pair)
    parser.add_argument("--consistency_weight", type=float, default=0.0, help="Start with 0.0; increase only after alignment is stable.")
    parser.add_argument("--consistency_kind", type=str, default="kl", choices=["mse", "kl"])

    # Loader / split
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_fraction", type=float, default=0.2)

    # Stage 1: AE training
    parser.add_argument("--train_autoencoders", action="store_true")
    parser.add_argument("--ae_epochs", type=int, default=10)
    parser.add_argument("--ae_lr", type=float, default=1e-4)
    parser.add_argument("--ae_weight_decay", type=float, default=1e-5)

    # Load AE checkpoints
    parser.add_argument("--load_best_ae_clean", action="store_true")
    parser.add_argument("--load_best_ae_noisy", action="store_true")
    parser.add_argument("--ae_clean_ckpt_path", type=str, default=None)
    parser.add_argument("--ae_noisy_ckpt_path", type=str, default=None)

    # Stage 2
    parser.add_argument("--train_stage2", action="store_true")
    parser.add_argument("--train_noisy_proj_only", action="store_true", help="Skip clean risk training; train ONLY proj_noisy on anchors.")
    parser.add_argument("--stage2_epochs", type=int, default=10)
    parser.add_argument("--stage2_lr", type=float, default=1e-4)
    parser.add_argument("--stage2_weight_decay", type=float, default=1e-5)

    # Load pretrained risk head (and optionally proj_clean) for proj-only training
    parser.add_argument("--load_risk_head", action="store_true")
    parser.add_argument("--risk_head_ckpt_path", type=str, default=None)

    # Load pretrained proj_noisy (optional)
    parser.add_argument("--load_proj_noisy", action="store_true")
    parser.add_argument("--proj_noisy_ckpt_path", type=str, default=None)

    # Eval
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--save_annotations", action="store_true")

    # W&B / sweep
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--wandb", action="store_true")

    # Output / misc
    parser.add_argument("--best_model_path", type=str, default="./models/risk_predictor/best_model.pt")
    parser.add_argument("--output_root", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_config", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)

    args = parser.parse_args()
    args = apply_yaml_overrides(parser, args)

    if args.load_best_ae_clean and not args.ae_clean_ckpt_path:
        raise SystemExit("--load_best_ae_clean requires --ae_clean_ckpt_path")
    if args.load_best_ae_noisy and not args.ae_noisy_ckpt_path:
        raise SystemExit("--load_best_ae_noisy requires --ae_noisy_ckpt_path")

    if args.load_risk_head and not args.risk_head_ckpt_path:
        raise SystemExit("--load_risk_head requires --risk_head_ckpt_path")

    if args.load_proj_noisy and not args.proj_noisy_ckpt_path:
        raise SystemExit("--load_proj_noisy requires --proj_noisy_ckpt_path")

    run_task(args)