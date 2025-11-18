import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import os
import numpy as np
from tqdm import tqdm
import sys

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

        dataset = EpisodeDataset(
            root_dir=f"./data/raw/{args.dataset}/frames",
            labels_json_path=f"./data/frame_targets/{args.dataset}/targets.json",
            transform=train_transform,
        )

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

        dataset = EpisodeDataset(
            root_dir=f"./data/raw/{args.dataset}/frames",
            labels_json_path=f"./data/frame_targets/{args.dataset}/targets.json",
            transform=train_transform,
        )
        model = MultiTaskClipModel(
            num_classes_per_task=args.num_classes_per_task,
            z_dim=args.z_dim,
            encoder_name=args.encoder_name,
            pretrained=False,  # weights will be loaded from checkpoint
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train, evaluate, and extract embeddings for frame encoder."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="L2D",
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
        default=20,
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

    # Paths, seed, outputs
    parser.add_argument(
        "--best_model_path",
        type=str,
        default="./checkpoints/best_model.pt",
        help="Path to save / load the best model checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./embeddings_out",
        help="Directory to save extracted embeddings.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for numpy and torch.",
    )

    args = parser.parse_args()

    if args.task_weights is not None:
        if len(args.task_weights) != len(args.num_classes_per_task):
            raise ValueError(
                f"task_weights length {len(args.task_weights)} does not match "
                f"num_classes_per_task length {len(args.num_classes_per_task)}"
            )

    run_task(args)
