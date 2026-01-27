import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from scipy.stats import spearmanr
import wandb
from sklearn.metrics import (mean_absolute_error,cohen_kappa_score,mean_squared_error,r2_score,confusion_matrix,)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.graph_encoding.data_loaders import get_graph_dataset
from src.graph_encoding.autoencoder import (HeteroGraphAutoencoder,batched_graph_embeddings,QuantileFeatureQuantizer,
                                            feature_loss,edge_loss,)
from src.graph_encoding.risk_prediction import RiskPredictionHead
from src.experiment_utils import (set_seed,seed_worker,risk_to_class_safe,infer_graph_emb_dim,apply_yaml_overrides,
                                  resolve_paths,_format_confusion_matrix,_print_access,_require_dir,_require_file,
                                  _make_fallback_run_id,_ensure_dir,classification_metrics_from_cm,log_annotations,)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_task(args: argparse.Namespace):
    is_eval_only_flag = bool(args.evaluate) and not bool(args.train_autoencoder) and not bool(args.train_risk)
    if is_eval_only_flag:
        original_best_model_path = args.best_model_path
        original_wandb = args.wandb
        _require_file(original_best_model_path, "Checkpoint (--best_model_path) for evaluation")
        ckpt = torch.load(original_best_model_path, map_location="cpu", weights_only=False)
        if "args" in ckpt:
            print("[info] Overwriting CLI args with args from checkpoint for model compatibility.")
            saved_args_dict = ckpt["args"]
            for k, v in saved_args_dict.items():
                if hasattr(args, k):
                    setattr(args, k, v)
            # Restore essential CLI args that should not be overwritten by the checkpoint's args
            args.evaluate = True
            args.train_autoencoder = False
            args.train_risk = False
            args.best_model_path = original_best_model_path
            args.wandb = original_wandb
        else:
            print("[warn] Checkpoint does not contain 'args' dictionary. Using CLI args, which may lead to errors.")

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

    # ---------------- CLI / seed / dataset enforcement ----------------
    set_seed(args.seed)
    
    if args.l2d:
        args.data_root = os.path.join("data", "L2D")
    elif args.nup:
        args.data_root = os.path.join("data", "NuPlan")
    
    args.data_root = os.path.abspath(args.data_root)

    dataset_name = ""
    if args.clean:
        dataset_name = "clean"
    elif args.noisy is not None:
        dataset_name = f"noisy_{args.noisy}"

    side_info_str = "_with_side_info" if args.with_side_information else ""
    pred_tag = "_class" if args.prediction_mode == "classification" else "_reg"

    # ---------------- OUTPUT ISOLATION (NO OVERWRITES) ----------------
    if getattr(args, "run_id", None):
        run_id = args.run_id

    is_eval_only = bool(args.evaluate) and not bool(args.train_autoencoder) and not bool(args.train_risk)

    if is_eval_only:
        # EVAL-ONLY: use the provided checkpoint path as-is.
        best_model_path = os.path.abspath(args.best_model_path)
        _require_file(best_model_path, "Checkpoint (--best_model_path) for evaluation")
        run_dir = os.path.dirname(best_model_path)
        ae_ckpt_path = best_model_path.replace("_best_model.pt", "_ae_best_model.pt")
        eval_results_path = os.path.join(run_dir, "evaluation_results.json")
        if wandb_run is not None:
            wandb.config.update(
                {"run_id": run_id, "run_dir": run_dir, "best_model_path": best_model_path},
                allow_val_change=True,
            )
    else:
        # If a specific path is given (and it's not the default placeholder), use it directly.
        if args.best_model_path and args.best_model_path != "./models/risk_predictor/best_model.pt":
            best_model_path = os.path.abspath(args.best_model_path)
            run_dir = os.path.dirname(best_model_path)
            _ensure_dir(run_dir)
        else:
            # Original path-building logic
            output_root = _ensure_dir(os.path.abspath(args.output_root))
            run_dir = _ensure_dir(
                os.path.join(
                    output_root,
                    "4Ba",
                    dataset_name,
                    ("classification" if args.prediction_mode == "classification" else "regression"),
                    run_id,
                )
            )
            if os.path.basename(args.best_model_path) == "best_model.pt":
                risk_ckpt_name = f"4Ba_{dataset_name}{side_info_str}{pred_tag}_best_model.pt"
            else:
                risk_ckpt_name = os.path.basename(args.best_model_path)
            best_model_path = os.path.join(run_dir, risk_ckpt_name)
        
        ae_ckpt_path = best_model_path.replace("_best_model.pt", "_ae_best_model.pt")
        eval_results_path = os.path.join(run_dir, "evaluation_results.json")
        if wandb_run is not None:
            wandb.config.update({"run_id": run_id, "run_dir": run_dir}, allow_val_change=True)

    paths = resolve_paths(args, dataset_name)

    # ---------------- dataset paths ----------------
    _print_access("TRAIN graphs root", paths["train_graph_root"])
    _print_access("TRAIN risk scores", paths["train_risk_path"])
    _require_dir(paths["train_graph_root"], "Training graphs")
    _require_file(paths["train_risk_path"], "Training risk scores JSON")

    side_information_path = None

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

    # Deterministic split (same split used for AE training and risk-head training)
    val_size = int(args.val_fraction * len(train_dataset_full))
    train_size = len(train_dataset_full) - val_size
    split_gen = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size], generator=split_gen)

    loader_gen = torch.Generator().manual_seed(args.seed)
    common_loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=loader_gen,
    )

    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_kwargs)

    # ---------------- models ----------------
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
        num_decoder_layers=args.num_decoder_layers,
        activation=args.activation,
        dropout_rate=args.dropout_rate,
        side_info_dim=side_info_dim,
    ).to(device)

    # Infer graph embedding dim once (after encoder exists)
    graph_emb_dim = infer_graph_emb_dim(
        encoder, quantizer, train_loader, metadata, embed_dim_per_type=args.embed_dim
    )
    print(f"[info] inferred graph_emb_dim = {graph_emb_dim}")

    # Risk head
    output_dim = 1 if args.prediction_mode == "regression" else args.num_classes
    prediction_head = RiskPredictionHead(
        input_dim=graph_emb_dim,
        hidden_dim=args.risk_hidden_dim,
        output_dim=output_dim,
        mode=args.prediction_mode,
    ).to(device)

    risk_loss_fn = nn.MSELoss() if args.prediction_mode == "regression" else nn.CrossEntropyLoss()

    def encode_graph_embeddings(batch):
        batch = quantizer.transform_inplace(batch).to(device)
        z_dict, feat_logits, edge_logits = encoder(batch)
        g = batched_graph_embeddings(z_dict, batch, metadata, embed_dim_per_type=args.embed_dim)
        return batch, z_dict, feat_logits, edge_logits, g

    # ---------------- Stage 1: train AE (reconstruction) ----------------
    best_ae_val = float("inf")

    if args.train_autoencoder:
        print(f"Stage 1/2: Training AUTOENCODER (reconstruction only) on {dataset_name}...")

        ae_opt = torch.optim.Adam(
            encoder.parameters(),
            lr=args.ae_lr,
            weight_decay=args.ae_weight_decay,
        )

        for epoch in range(1, args.ae_epochs + 1):
            encoder.train()
            total = 0.0
            n_batches = 0

            for batch in tqdm(train_loader, desc=f"AE Epoch {epoch:02d} [train]"):
                ae_opt.zero_grad()
                batch, z_dict, feat_logits, edge_logits, _g = encode_graph_embeddings(batch)

                # Reconstruction loss = feature + edge
                l_feat = feature_loss(feat_logits, batch)
                l_edge = edge_loss(edge_logits, z_dict, encoder.edge_decoders)
                loss = l_feat + l_edge

                loss.backward()
                ae_opt.step()

                total += float(loss.item())
                n_batches += 1

            train_loss = total / max(n_batches, 1)

            # val
            encoder.eval()
            vtotal = 0.0
            vfeat = 0.0
            vedge = 0.0
            vn = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"AE Epoch {epoch:02d} [val]"):
                    batch, z_dict, feat_logits, edge_logits, _g = encode_graph_embeddings(batch)
                    l_feat = feature_loss(feat_logits, batch)
                    l_edge = edge_loss(edge_logits, z_dict, encoder.edge_decoders)
                    loss = l_feat + l_edge

                    vtotal += float(loss.item())
                    vfeat += float(l_feat.item())
                    vedge += float(l_edge.item())
                    vn += 1

            val_loss = vtotal / max(vn, 1)
            val_feat = vfeat / max(vn, 1)
            val_edge = vedge / max(vn, 1)

            print(
                f"AE Epoch {epoch:02d} | train_recon={train_loss:.4f} | "
                f"val_recon={val_loss:.4f} | val_feat={val_feat:.4f} | val_edge={val_edge:.4f}"
            )

            if wandb_run is not None:
                wandb.log(
                    {
                        "ae/epoch": epoch,
                        "ae/train_recon": train_loss,
                        "ae/val_recon": val_loss,
                        "ae/val_feat": val_feat,
                        "ae/val_edge": val_edge,
                    }
                )

            if val_loss < best_ae_val:
                best_ae_val = val_loss
                _ensure_dir(os.path.dirname(ae_ckpt_path))
                torch.save(
                    {
                        "stage": "autoencoder_best",
                        "encoder_state_dict": encoder.state_dict(),
                        "best_ae_val_recon": best_ae_val,
                        "graph_emb_dim": graph_emb_dim,
                        "dataset_name": dataset_name,
                        "args": dict(vars(args)),
                        "run_id": run_id,
                        "run_dir": run_dir,
                    },
                    ae_ckpt_path,
                )
                print(f"  -> New best AE saved to {ae_ckpt_path}")

    # Optionally load best AE ckpt before risk training/eval
    if args.load_best_ae:
        _require_file(ae_ckpt_path, "AE checkpoint (--load_best_ae expects *_ae_best_model.pt)")
        ck = torch.load(ae_ckpt_path, map_location=device, weights_only=False)
        encoder.load_state_dict(ck["encoder_state_dict"])
        encoder.eval()
        print(f"[ok] loaded AE checkpoint: {ae_ckpt_path}")

    # Freeze encoder for stage 2
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    # ---------------- Stage 2: train risk head (encoder frozen) ----------------
    best_val_loss = float("inf")

    if args.train_risk:
        print(f"Stage 2/2: Training RISK HEAD (encoder frozen) on {dataset_name}...")

        risk_opt = torch.optim.Adam(
            prediction_head.parameters(),
            lr=args.risk_lr,
            weight_decay=args.risk_weight_decay,
        )

        for epoch in range(1, args.risk_epochs + 1):
            prediction_head.train()
            total = 0.0
            correct = 0
            seen = 0

            for batch in tqdm(train_loader, desc=f"Risk Epoch {epoch:02d} [train]"):
                risk_opt.zero_grad()
                batch, _z, _feat_logits, _edge_logits, g = encode_graph_embeddings(batch)
                pred = prediction_head(g)

                if args.prediction_mode == "regression":
                    target = batch.y.view(-1, 1).float()
                    loss = risk_loss_fn(pred, target)
                else:
                    target = risk_to_class_safe(batch.y)
                    loss = risk_loss_fn(pred, target)
                    pred_cls = torch.argmax(pred, dim=-1)
                    correct += int((pred_cls == target).sum().item())
                    seen += int(target.numel())

                loss.backward()
                risk_opt.step()
                total += float(loss.item())

            train_loss = total / max(len(train_loader), 1)
            train_acc = (correct / max(seen, 1)) if args.prediction_mode == "classification" else None

            # val
            prediction_head.eval()
            vtotal = 0.0
            vcorrect = 0
            vseen = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Risk Epoch {epoch:02d} [val]"):
                    batch, _z, _feat_logits, _edge_logits, g = encode_graph_embeddings(batch)
                    pred = prediction_head(g)

                    if args.prediction_mode == "regression":
                        target = batch.y.view(-1, 1).float()
                        vtotal += float(risk_loss_fn(pred, target).item())
                    else:
                        target = risk_to_class_safe(batch.y)
                        vtotal += float(risk_loss_fn(pred, target).item())
                        pred_cls = torch.argmax(pred, dim=-1)
                        vcorrect += int((pred_cls == target).sum().item())
                        vseen += int(target.numel())

            val_loss = vtotal / max(len(val_loader), 1)
            val_acc = (vcorrect / max(vseen, 1)) if args.prediction_mode == "classification" else None

            if args.prediction_mode == "classification":
                print(
                    f"Risk Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
                    f"| train_acc={train_acc:.4f} | val_acc={val_acc:.4f}"
                )
            else:
                print(f"Risk Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

            if wandb_run is not None:
                payload = {"risk/epoch": epoch, "risk/train_loss": train_loss, "risk/val_loss": val_loss}
                if args.prediction_mode == "classification":
                    payload.update({"risk/train_acc": train_acc, "risk/val_acc": val_acc})
                wandb.log(payload)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _ensure_dir(os.path.dirname(best_model_path))
                torch.save(
                    {
                        "stage": "risk_best",
                        "encoder_state_dict": encoder.state_dict(),  # frozen AE
                        "prediction_head_state_dict": prediction_head.state_dict(),
                        "best_val_loss": best_val_loss,
                        "graph_emb_dim": graph_emb_dim,
                        "dataset_name": dataset_name,
                        "args": dict(vars(args)),
                        "run_id": run_id,
                        "run_dir": run_dir,
                    },
                    best_model_path,
                )
                print(f"  -> New best 4Ba model saved to {best_model_path}")

    # ---------------- EVALUATE ----------------
    if args.evaluate:
        print(f"Evaluating 4Ba (AE frozen + risk head) on {dataset_name}...")

        _require_file(best_model_path, "Checkpoint (--best_model_path)")
        ckpt = torch.load(best_model_path, map_location=device, weights_only=False)

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
            side_information_path=None,
            risk_scores_path=paths["eval_risk_path"],
        )

        eval_loader = DataLoader(
            eval_dataset_full,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker if args.num_workers > 0 else None,
        )

        y_true_all, y_pred_all = [], []
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"[eval:{dataset_name}]"):
                batch, _z, _feat_logits, _edge_logits, g = encode_graph_embeddings(batch)
                pred = prediction_head(g)

                if args.prediction_mode == "regression":
                    target = batch.y.view(-1, 1).float()
                    loss = risk_loss_fn(pred, target)
                    y_true_all.append(target.cpu().numpy())
                    y_pred_all.append(pred.cpu().numpy())
                else:
                    target = risk_to_class_safe(batch.y)
                    loss = risk_loss_fn(pred, target)
                    pred_cls = torch.argmax(pred, dim=-1)
                    y_true_all.append(target.cpu().numpy())
                    y_pred_all.append(pred_cls.cpu().numpy())

                total_loss += float(loss.item())
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        y_true_np = np.concatenate(y_true_all)
        y_pred_np = np.concatenate(y_pred_all)

        metrics = {"eval/loss": avg_loss}
        if args.prediction_mode == "classification":
            cm = confusion_matrix(y_true_np, y_pred_np, labels=range(args.num_classes))
            cls_metrics = classification_metrics_from_cm(cm)

            cls_metrics["ordinal_mae_bins"] = mean_absolute_error(y_true_np, y_pred_np)
            cls_metrics["qwk"] = cohen_kappa_score(y_true_np, y_pred_np, weights='quadratic')

            metrics.update(cls_metrics)
            cm_list = metrics.pop("confusion_matrix")

            print(f"\n========== {dataset_name} evaluation results ==========")
            for k, v in sorted(metrics.items()):
                 if isinstance(v, list):
                    print(f"{k}: {v}")
                 else:
                    print(f"{k}: {v:.4f}")

            print("\nConfusion matrix (rows=true, cols=pred):")
            print(_format_confusion_matrix(np.array(cm_list)))
            print("===============================================\n")

        else:
            metrics["eval/mae"] = mean_absolute_error(y_true_np, y_pred_np)
            metrics["eval/rmse"] = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
            metrics["eval/r2"] = r2_score(y_true_np, y_pred_np)
            rho, p = spearmanr(y_true_np, y_pred_np)
            metrics["eval/spearman_rho"] = rho
            metrics["eval/spearman_p"] = p

            print(f"{dataset_name} evaluation avg loss: {avg_loss:.4f}")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")

        if wandb_run is not None:
            wandb.log(metrics)

        if args.save_annotations:
            results = {}
            if os.path.exists(eval_results_path):
                with open(eval_results_path, "r") as f:
                    results = json.load(f)

            results.setdefault("4Ba", {})
            key = dataset_name
            if args.with_side_information:
                key += "_with_side_info"

            # Update results with all new metrics
            results["4Ba"][key] = avg_loss
            results["4Ba"][key + "_run_id"] = run_id
            results["4Ba"][key + "_run_dir"] = run_dir
            for k, v in metrics.items():
                results["4Ba"][key + "_" + k.replace("eval/", "")] = v

            with open(eval_results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved evaluation results to {eval_results_path}")

    if wandb_run is not None:
        wandb_run.finish()

    log_annotations(
        file_path="./experiment_results/df.csv",
        script_name="BaselineB",
        anchor_pct=0,
        noise_pct=args.noisy if args.noisy is not None else 0,
        seed=args.seed,
        metrics=metrics,
        domain="clean" if args.clean else "noisy",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4Ba: Autoencoder (reconstruction) then risk head (encoder frozen).")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--l2d", action="store_true", help="Use L2D dataset.")
    group.add_argument("--nup", action="store_true", help="Use NuPlan dataset.")
    
    parser.add_argument("--data_root", type=str, help="Root directory of the dataset.")


    # Dataset select
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--clean", action="store_true", help="Use clean data.")
    dataset_group.add_argument("--noisy", type=int, help="Use noisy data with a specified percentage (e.g., 10, 20).")

    parser.add_argument("--with_side_information", action="store_true")

    # Model
    parser.add_argument("--mode", type=str, default="all")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--num_encoder_layers", type=int, default=1)
    parser.add_argument("--num_decoder_layers", type=int, default=1)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--dropout_rate", type=float, default=0.1)

    # Risk head
    parser.add_argument("--risk_hidden_dim", type=int, default=64)
    parser.add_argument("--prediction_mode", type=str, default="classification", choices=["regression", "classification"])
    parser.add_argument("--num_classes", type=int, default=4)

    # Loader / split
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--quant_bins", type=int, default=32)

    # Stage 1: AE training
    parser.add_argument("--train_autoencoder", action="store_true")
    parser.add_argument("--ae_epochs", type=int, default=10)
    parser.add_argument("--ae_lr", type=float, default=1e-4)
    parser.add_argument("--ae_weight_decay", type=float, default=1e-5)
    parser.add_argument("--load_best_ae", action="store_true", help="Load *_ae_best_model.pt before training risk/eval")

    # Stage 2: risk training
    parser.add_argument("--train_risk", action="store_true")
    parser.add_argument("--risk_epochs", type=int, default=10)
    parser.add_argument("--risk_lr", type=float, default=1e-4)
    parser.add_argument("--risk_weight_decay", type=float, default=1e-5)

    # Eval
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--save_annotations", action="store_true")

    # W&B / sweep
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")

    # Output / misc
    parser.add_argument("--best_model_path", type=str, default="./models/risk_predictor/best_model.pt")
    parser.add_argument("--output_root", type=str, default="./outputs", help="Root dir for per-run outputs (no overwrites).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_config", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None, help="Optional: force a specific run_id (prevents random local-* ids).")

    args = parser.parse_args()
    args = apply_yaml_overrides(parser, args)
    run_task(args)
