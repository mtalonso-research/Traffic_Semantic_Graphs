import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import os
import numpy as np
from tqdm import tqdm
import sys
import wandb  
import yaml 
from torch.utils.data import ConcatDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.frame_encoding.frame_targets import extract_targets
from src.frame_encoding.data_loader import EpisodeDataset, collate_episodes
from src.frame_encoding.frame_encoder import (
    MultiTaskClipModel,
    train_one_epoch,
    evaluate,
    encode_batch_episodes,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_task(args):

    # --- wandb init / sweeps ---
    wandb_run = None
    if args.use_wandb and args.wandb_mode != "disabled" and args.train_encoder:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name or None,
            group=args.wandb_group or None,
            mode=args.wandb_mode,
            config=vars(args),
        )
        # Let sweeps override CLI arguments
        cfg = wandb.config
        for k, v in cfg.items():
            setattr(args, k, v)

    if args.train_encoder:
        print("Final args used for this run:")
        for k, v in sorted(vars(args).items()):
            print(f"  {k}: {v}")

    # After possible override, use args as the single source of truth
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.extract_targets:
        extract_targets(
            graph_dir=f'./data/graphical_final/{args.dataset}/',
            tag_dir=f'./data/semantic_tags/{args.dataset}/',
            output_dir=f'./data/frame_targets/{args.dataset}/'
        )

    if args.train_encoder:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # (C,H,W) in [0,1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        if args.dataset == 'L2D':
            dataset = EpisodeDataset(
                root_dir=f"./data/raw/{args.dataset}/frames",
                labels_json_path=f"./data/frame_targets/{args.dataset}/targets.json",
                transform=train_transform,
                episode_dir_format = "Episode{episode_id:06d}/observation.images.front_left",
                domain_name="L2D",
            )
        elif args.dataset == 'NuPlan':
            dataset = EpisodeDataset(
                root_dir=f"./data/raw/{args.dataset}/frames",
                labels_json_path=f"./data/frame_targets/{args.dataset}/targets.json",
                transform=train_transform,
                frames_per_sample=10,
                samples_per_episode=10,
                min_frame_gap=30,
                domain_name="NuPlan",
            )
        elif args.dataset == 'mix':
            l2d_dataset = EpisodeDataset(
                root_dir=f"./data/raw/L2D/frames",
                labels_json_path=f"./data/frame_targets/L2D/targets.json",
                transform=train_transform,
                episode_dir_format = "Episode{episode_id:06d}/observation.images.front_left",
                domain_name="L2D",
            )
            nup_dataset = EpisodeDataset(
                root_dir=f"./data/raw/NuPlan/frames",
                labels_json_path=f"./data/frame_targets/NuPlan/targets.json",
                transform=train_transform,
                frames_per_sample=10,
                samples_per_episode=4,
                min_frame_gap=30,
                domain_name="NuPlan",
            )

            num_l2d = len(l2d_dataset)
            num_nup = len(nup_dataset)
            target_l2d = min(num_l2d, num_nup)
            rng = np.random.default_rng(args.seed)
            l2d_indices = np.arange(num_l2d)
            rng.shuffle(l2d_indices)
            l2d_train_indices = l2d_indices[:target_l2d]
            l2d_subset = Subset(l2d_dataset, l2d_train_indices)
            dataset = ConcatDataset([l2d_subset, nup_dataset])
            
        else: print('DID NOT RECOGNIZE THE DATASET!!')

        rng = np.random.default_rng(args.seed)
        indices = np.arange(len(dataset))
        rng.shuffle(indices)

        val_size = int(len(indices) * args.val_fraction)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        print(f"Train episodes: {len(train_dataset)}, Val episodes: {len(val_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_episodes,
            pin_memory=args.pin_memory,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_episodes,
            pin_memory=args.pin_memory,
        )

        model = MultiTaskClipModel(
            num_classes_per_task=args.num_classes_per_task,
            z_dim=args.z_dim,
            encoder_name=args.encoder_name,
            pretrained=not args.no_pretrained,
            # NEW: frame head config
            frame_head_hidden_dims=args.frame_head_hidden_dims,
            frame_head_activation=args.frame_head_activation,
            frame_head_dropout=args.frame_head_dropout,
            # NEW: pooling config
            pooling=args.pooling,
            num_pool_heads=args.num_pool_heads,
            pool_hidden_dim=args.pool_hidden_dim,
            pool_dropout=args.pool_dropout,
            pool_temperature_init=args.pool_temperature_init,
            # NEW: task head config
            task_head_hidden_dims=args.task_head_hidden_dims,
            task_head_activation=args.task_head_activation,
            task_head_dropout=args.task_head_dropout,
            # NEW: encoder freezing
            encoder_trainable=args.encoder_trainable,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        best_val_loss = float("inf")

        for epoch in range(1, args.num_epochs + 1):
            print(f'Running Epoch {epoch} ...')

            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                label_smoothing=args.label_smoothing,
                task_weights=args.task_weights,
            )

            val_loss, acc_per_task, mean_acc = evaluate(
                model,
                val_loader,
                device,
                label_smoothing=args.label_smoothing,
                task_weights=args.task_weights,
            )

            acc_str = ", ".join([f"task{k}: {acc:.3f}" for k, acc in enumerate(acc_per_task)])

            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"mean_acc={mean_acc:.3f} | "
                f"{acc_str}"
            )

            # --- wandb logging ---
            if wandb_run is not None:
                log_dict = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "mean_acc": mean_acc,
                    "best_val_loss": best_val_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                for k, acc in enumerate(acc_per_task):
                    log_dict[f"task_acc_{k}"] = acc
                wandb.log(log_dict, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(args.best_model_path), exist_ok=True)
                torch.save(model.state_dict(), args.best_model_path)
                print(f"  -> New best model saved to {args.best_model_path}")

    if args.evaluate:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # (C,H,W) in [0,1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        if args.dataset == 'L2D':
            dataset = EpisodeDataset(
                root_dir=f"./data/raw/{args.dataset}/frames",
                labels_json_path=f"./data/frame_targets/{args.dataset}/targets.json",
                transform=train_transform,
                episode_dir_format = "Episode{episode_id:06d}/observation.images.front_left",
                domain_name="L2D",
            )
        elif args.dataset == 'NuPlan':
            dataset = EpisodeDataset(
                root_dir=f"./data/raw/{args.dataset}/frames",
                labels_json_path=f"./data/frame_targets/{args.dataset}/targets.json",
                transform=train_transform,
                frames_per_sample=10,
                samples_per_episode=10,
                min_frame_gap=30,
                domain_name="NuPlan",
            )
        elif args.dataset == 'mix':
            l2d_dataset = EpisodeDataset(
                root_dir=f"./data/raw/L2D/frames",
                labels_json_path=f"./data/frame_targets/L2D/targets.json",
                transform=train_transform,
                episode_dir_format = "Episode{episode_id:06d}/observation.images.front_left",
                domain_name="L2D",
            )
            nup_dataset = EpisodeDataset(
                root_dir=f"./data/raw/NuPlan/frames",
                labels_json_path=f"./data/frame_targets/NuPlan/targets.json",
                transform=train_transform,
                frames_per_sample=10,
                samples_per_episode=4,
                min_frame_gap=30,
                domain_name="NuPlan",
            )
            dataset = ConcatDataset([l2d_dataset, nup_dataset])
        else: print('DID NOT RECOGNIZE THE DATASET!!')

        model = MultiTaskClipModel(
            num_classes_per_task=args.num_classes_per_task,
            z_dim=args.z_dim,
            encoder_name=args.encoder_name,
            pretrained=not args.no_pretrained,
            # NEW: frame head config
            frame_head_hidden_dims=args.frame_head_hidden_dims,
            frame_head_activation=args.frame_head_activation,
            frame_head_dropout=args.frame_head_dropout,
            # NEW: pooling config
            pooling=args.pooling,
            num_pool_heads=args.num_pool_heads,
            pool_hidden_dim=args.pool_hidden_dim,
            pool_dropout=args.pool_dropout,
            pool_temperature_init=args.pool_temperature_init,
            # NEW: task head config
            task_head_hidden_dims=args.task_head_hidden_dims,
            task_head_activation=args.task_head_activation,
            task_head_dropout=args.task_head_dropout,
            # NEW: encoder freezing
            encoder_trainable=args.encoder_trainable,
        ).to(device)

        print(f"Loading model from {args.best_model_path}")
        state = torch.load(args.best_model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        full_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_episodes,
            pin_memory=args.pin_memory,
        )

        val_loss, acc_per_task, mean_acc = evaluate(
            model,
            full_loader,
            device,
            label_smoothing=args.label_smoothing,
            task_weights=args.task_weights,
        )

        print(f"Evaluation results:")
        print(f"  val_loss = {val_loss:.4f}")
        for k, acc in enumerate(acc_per_task):
            print(f"  task{k}_acc = {acc:.4f}")
        print(f"  mean_acc = {mean_acc:.4f}")

        clip_embeddings = {}   # episode_id -> (z_dim,)
        frame_embeddings = {}  # episode_id -> (T_i, z_dim)

        with torch.no_grad():
            for frames_list, targets_batch, episode_ids in tqdm(full_loader, desc="Extracting embeddings"):
                z_split, clip_embs = encode_batch_episodes(model, frames_list)

                clip_embs_cpu = clip_embs.cpu()
                z_split_cpu = [z.cpu() for z in z_split]

                for eid, z_frames, z_clip in zip(episode_ids, z_split_cpu, clip_embs_cpu):
                    clip_embeddings[eid] = z_clip      # (z_dim,)
                    frame_embeddings[eid] = z_frames    # (T_i, z_dim)

        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(clip_embeddings, os.path.join(args.output_dir, "clip_embeddings.pt"))
        torch.save(frame_embeddings, os.path.join(args.output_dir, "frame_embeddings.pt"))

        print(f"Saved clip and frame embeddings to {args.output_dir}/")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train, evaluate, and extract embeddings for frame encoder."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="null",
        help="Dataset name (used in paths under ./data/).",
    )
    parser.add_argument(
        "--extract_targets",
        action="store_true",
        default=False,
        help="Whether to run target extraction from graphs and semantic tags.",
    )
    parser.add_argument(
        "--train_encoder",
        action="store_true",
        default=False,
        help="Whether to train the frame encoder.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="Whether to run evaluation / embedding extraction.",
    )

    # Model / training hyperparams
    parser.add_argument(
        "--num_classes_per_task",
        nargs="+",
        type=int,
        default=[5, 5, 5, 5, 5],
        help="List of number of classes per task, e.g. --num_classes_per_task 5 5 5 5 5",
    )
    parser.add_argument(
        "--z_dim",
        type=int,
        default=256,
        help="Dimension of clip embedding.",
    )
    parser.add_argument(
        "--encoder_name",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50"],
        help="Backbone encoder architecture.",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        default=False,
        help="If set, do NOT use ImageNet pretrained weights.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (episodes per batch).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=False,
        help="Use pin_memory in DataLoader (useful on CUDA).",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for Adam optimizer.",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.05,
        help="Label smoothing value for multi-task loss.",
    )
    parser.add_argument(
        "--task_weights",
        nargs="*",
        type=float,
        default=None,
        help="Optional per-task weights, e.g. --task_weights 1.0 1.0 0.5 1.0 1.0",
    )

    # NEW: model architecture hyperparams
    parser.add_argument(
        "--frame_head_hidden_dims",
        nargs="+",
        type=int,
        default=[512],
        help="Hidden dims for frame head MLP, e.g. --frame_head_hidden_dims 512 256",
    )
    parser.add_argument(
        "--frame_head_activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "tanh", "sigmoid"],
        help="Activation for frame head MLP.",
    )
    parser.add_argument(
        "--frame_head_dropout",
        type=float,
        default=0.0,
        help="Dropout for frame head MLP.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "max", "meanmax", "attn", "multihead_attn", "lse", "gated"],
        help="Temporal pooling strategy.",
    )
    parser.add_argument(
        "--num_pool_heads",
        type=int,
        default=4,
        help="Number of heads for multihead_attn pooling.",
    )
    parser.add_argument(
        "--pool_hidden_dim",
        type=int,
        default=128,
        help="Hidden dim for gated pooling.",
    )
    parser.add_argument(
        "--pool_dropout",
        type=float,
        default=0.0,
        help="Dropout for attention/gated pooling.",
    )
    parser.add_argument(
        "--pool_temperature_init",
        type=float,
        default=1.0,
        help="Initial temperature for LSE pooling.",
    )
    parser.add_argument(
        "--task_head_hidden_dims",
        nargs="*",
        type=int,
        default=[],
        help="Hidden dims for task head MLP, empty means single linear layer.",
    )
    parser.add_argument(
        "--task_head_activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "tanh", "sigmoid"],
        help="Activation for task head MLP.",
    )
    parser.add_argument(
        "--task_head_dropout",
        type=float,
        default=0.0,
        help="Dropout for task head MLP.",
    )
    parser.add_argument(
        "--encoder_trainable",
        type=str,
        default="all",
        help='Encoder training mode: "all", "none", or "partial:N".',
    )

    # Paths, seed, outputs
    parser.add_argument(
        "--best_model_path",
        type=str,
        default="./models/frame_encoder/best_model.pt",
        help="Path to save / load the best model checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/frame_embeddings/L2D",
        help="Directory to save extracted embeddings.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for numpy and torch.",
    )

    # NEW: wandb args
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging and sweeps.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="wandb project name.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="wandb entity (user or team).",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional wandb run name.",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="Optional group name for this run.",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="wandb mode.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config file to override/default arguments.",
    )

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)

        for k, v in cfg.items():
            if not hasattr(args, k):
                print(f"Warning: key '{k}' in config not found in argparse args; ignoring.")
                continue

            default_val = parser.get_default(k)
            current_val = getattr(args, k)

            if current_val == default_val:
                setattr(args, k, v)

    if args.task_weights is not None:
        if len(args.task_weights) != len(args.num_classes_per_task):
            raise ValueError(
                f"task_weights length {len(args.task_weights)} does not match "
                f"num_classes_per_task length {len(args.num_classes_per_task)}"
            )

    run_task(args)
