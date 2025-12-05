import argparse
import torch
from torchvision import transforms
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
import os
import numpy as np
from tqdm import tqdm
import sys
import wandb  
import yaml 
from torch.utils.data import ConcatDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_encoding.data_loaders import get_graph_dataset
from src.graph_encoding.autoencoder import (
    HeteroGraphAutoencoder,
    feature_loss,
    edge_loss,
    QuantileFeatureQuantizer,
    ProjectionHead,
    kl_divergence_between_gaussians
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

    if args.train_encoder:
        l2d_dataset = get_graph_dataset(
            root_dir='./data/graphical_final/L2D/',
            mode=args.mode,
            side_information_path=args.side_info_path,
            node_features_to_exclude=args.node_features_to_exclude,
            risk_scores_path=args.risk_scores_path
        )
        nuplan_dataset = get_graph_dataset(
            root_dir='./data/graphical_final/nuplan_boston/', # Or other nuplan dataset
            mode=args.mode,
            side_information_path=args.side_info_path,
            node_features_to_exclude=args.node_features_to_exclude,
            risk_scores_path=args.risk_scores_path
        )

        l2d_quantizer = QuantileFeatureQuantizer(bins=32, node_types=l2d_dataset.get_metadata()[0])
        l2d_quantizer.fit(l2d_dataset)

        nuplan_quantizer = QuantileFeatureQuantizer(bins=32, node_types=nuplan_dataset.get_metadata()[0])
        nuplan_quantizer.fit(nuplan_dataset)

        # Split datasets into training and validation
        l2d_train_size = int((1 - args.val_fraction) * len(l2d_dataset))
        l2d_val_size = len(l2d_dataset) - l2d_train_size
        l2d_train_dataset, l2d_val_dataset = torch.utils.data.random_split(l2d_dataset, [l2d_train_size, l2d_val_size])

        nuplan_train_size = int((1 - args.val_fraction) * len(nuplan_dataset))
        nuplan_val_size = len(nuplan_dataset) - nuplan_train_size
        nuplan_train_dataset, nuplan_val_dataset = torch.utils.data.random_split(nuplan_dataset, [nuplan_train_size, nuplan_val_size])

        l2d_train_loader = DataLoader(
            l2d_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        nuplan_train_loader = DataLoader(
            nuplan_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

        l2d_val_loader = DataLoader(
            l2d_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        nuplan_val_loader = DataLoader(
            nuplan_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

        l2d_autoencoder = HeteroGraphAutoencoder(
            metadata=l2d_dataset.get_metadata(),
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            quantizer_spec=l2d_quantizer.spec(),
            feat_emb_dim=16,
            use_feature_mask=False,
            feature_entropy=None,
            trainable_gates=False,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            activation=args.activation,
            dropout_rate=args.dropout_rate,
            side_info_dim=args.side_info_dim if args.side_info_path else 0,
        ).to(device)

        nuplan_autoencoder = HeteroGraphAutoencoder(
            metadata=nuplan_dataset.get_metadata(),
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            quantizer_spec=nuplan_quantizer.spec(),
            feat_emb_dim=16,
            use_feature_mask=False,
            feature_entropy=None,
            trainable_gates=False,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            activation=args.activation,
            dropout_rate=args.dropout_rate,
            side_info_dim=args.side_info_dim if args.side_info_path else 0,
        ).to(device)
        
        projection_head = ProjectionHead(in_dim=args.embed_dim, proj_dim=args.embed_dim).to(device)

        optimizer = torch.optim.Adam(
            list(l2d_autoencoder.parameters()) + list(nuplan_autoencoder.parameters()) + list(projection_head.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        best_val_loss = float("inf")

        for epoch in range(1, args.num_epochs + 1):
            print(f'Running Epoch {epoch} ...')

            l2d_autoencoder.train()
            nuplan_autoencoder.train()
            projection_head.train()
            
            total_train_loss = 0
            
            # This assumes the datasets are of same length, which might not be true
            # A better implementation would be to use iterators
            train_loader = zip(l2d_train_loader, nuplan_train_loader)

            for l2d_data, nuplan_data in tqdm(train_loader, desc="Training"):
                l2d_data = l2d_quantizer.transform_inplace(l2d_data).to(device)
                nuplan_data = nuplan_quantizer.transform_inplace(nuplan_data).to(device)

                optimizer.zero_grad()

                # L2D forward pass
                l2d_z_dict, l2d_feat_logits, l2d_edge_logits = l2d_autoencoder(l2d_data)
                l2d_recon_loss = feature_loss(l2d_feat_logits, l2d_data) + edge_loss(l2d_edge_logits, l2d_z_dict, l2d_autoencoder.edge_decoders)

                # NuPlan forward pass
                nuplan_z_dict, nuplan_feat_logits, nuplan_edge_logits = nuplan_autoencoder(nuplan_data)
                nuplan_recon_loss = feature_loss(nuplan_feat_logits, nuplan_data) + edge_loss(nuplan_edge_logits, nuplan_z_dict, nuplan_autoencoder.edge_decoders)

                # Projection and KL divergence
                l2d_ego_z = l2d_z_dict['ego']
                nuplan_ego_z = nuplan_z_dict['ego']
                
                l2d_proj_z = projection_head(l2d_ego_z)
                nuplan_proj_z = projection_head(nuplan_ego_z)
                
                kl_loss = kl_divergence_between_gaussians(l2d_proj_z, nuplan_proj_z)

                loss = l2d_recon_loss + nuplan_recon_loss + args.kl_weight * kl_loss
                
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(l2d_train_loader)

            # Validation loop
            l2d_autoencoder.eval()
            nuplan_autoencoder.eval()
            projection_head.eval()
            
            total_val_loss = 0
            total_kl_div = 0
            total_l2d_recon_loss = 0
            total_nuplan_recon_loss = 0
            
            val_loader = zip(l2d_val_loader, nuplan_val_loader)

            with torch.no_grad():
                for l2d_data, nuplan_data in tqdm(val_loader, desc="Validation"):
                    l2d_data = l2d_quantizer.transform_inplace(l2d_data).to(device)
                    nuplan_data = nuplan_quantizer.transform_inplace(nuplan_data).to(device)

                    # L2D forward pass
                    l2d_z_dict, l2d_feat_logits, l2d_edge_logits = l2d_autoencoder(l2d_data)
                    l2d_recon_loss = feature_loss(l2d_feat_logits, l2d_data) + edge_loss(l2d_edge_logits, l2d_z_dict, l2d_autoencoder.edge_decoders)

                    # NuPlan forward pass
                    nuplan_z_dict, nuplan_feat_logits, nuplan_edge_logits = nuplan_autoencoder(nuplan_data)
                    nuplan_recon_loss = feature_loss(nuplan_feat_logits, nuplan_data) + edge_loss(nuplan_edge_logits, nuplan_z_dict, nuplan_autoencoder.edge_decoders)

                    # Projection and KL divergence
                    l2d_ego_z = l2d_z_dict['ego']
                    nuplan_ego_z = nuplan_z_dict['ego']
                    
                    l2d_proj_z = projection_head(l2d_ego_z)
                    nuplan_proj_z = projection_head(nuplan_ego_z)
                    
                    kl_loss = kl_divergence_between_gaussians(l2d_proj_z, nuplan_proj_z)

                    loss = l2d_recon_loss + nuplan_recon_loss + args.kl_weight * kl_loss
                    
                    total_val_loss += loss.item()
                    total_kl_div += kl_loss.item()
                    total_l2d_recon_loss += l2d_recon_loss.item()
                    total_nuplan_recon_loss += nuplan_recon_loss.item()

            avg_val_loss = total_val_loss / len(l2d_val_loader)
            avg_kl_div = total_kl_div / len(l2d_val_loader)
            avg_l2d_recon_loss = total_l2d_recon_loss / len(l2d_val_loader)
            avg_nuplan_recon_loss = total_nuplan_recon_loss / len(l2d_val_loader)

            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_loss={avg_val_loss:.4f} | "
                f"kl_div={avg_kl_div:.4f} | "
                f"l2d_recon_loss={avg_l2d_recon_loss:.4f} | "
                f"nuplan_recon_loss={avg_nuplan_recon_loss:.4f}"
            )
            
            # Save the model at the end of each epoch
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs(os.path.dirname(args.best_model_path), exist_ok=True)
                torch.save({
                    'l2d_autoencoder': l2d_autoencoder.state_dict(),
                    'nuplan_autoencoder': nuplan_autoencoder.state_dict(),
                    'projection_head': projection_head.state_dict(),
                }, args.best_model_path)
                print(f"  -> New best model saved to {args.best_model_path}")
            
            # Note: evaluation logic needs to be updated as well
            # For simplicity, we will skip evaluation for now.
    
    if args.evaluate:
        # L2D dataset
        l2d_dataset = get_graph_dataset(
            root_dir='./data/graphical_final/L2D/',
            mode=args.mode,
            side_information_path=args.side_info_path,
            node_features_to_exclude=args.node_features_to_exclude,
            risk_scores_path=args.risk_scores_path
        )

        l2d_quantizer = QuantileFeatureQuantizer(bins=32, node_types=l2d_dataset.get_metadata()[0])
        l2d_quantizer.fit(l2d_dataset)

        l2d_autoencoder = HeteroGraphAutoencoder(
            metadata=l2d_dataset.get_metadata(),
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            quantizer_spec=l2d_quantizer.spec(),
            feat_emb_dim=16,
            use_feature_mask=False,
            feature_entropy=None,
            trainable_gates=False,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            activation=args.activation,
            dropout_rate=args.dropout_rate,
            side_info_dim=args.side_info_dim if args.side_info_path else 0,
        ).to(device)

        # NuPlan dataset
        nuplan_dataset = get_graph_dataset(
            root_dir='./data/graphical_final/nuplan_boston/', # Or other nuplan dataset
            mode=args.mode,
            side_information_path=args.side_info_path,
            node_features_to_exclude=args.node_features_to_exclude,
            risk_scores_path=args.risk_scores_path
        )
        nuplan_quantizer = QuantileFeatureQuantizer(bins=32, node_types=nuplan_dataset.get_metadata()[0])
        nuplan_quantizer.fit(nuplan_dataset)

        nuplan_autoencoder = HeteroGraphAutoencoder(
            metadata=nuplan_dataset.get_metadata(),
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            quantizer_spec=nuplan_quantizer.spec(),
            feat_emb_dim=16,
            use_feature_mask=False,
            feature_entropy=None,
            trainable_gates=False,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            activation=args.activation,
            dropout_rate=args.dropout_rate,
            side_info_dim=args.side_info_dim if args.side_info_path else 0,
        ).to(device)

        print(f"Loading models from {args.best_model_path}")
        state = torch.load(args.best_model_path, map_location=device)
        l2d_autoencoder.load_state_dict(state['l2d_autoencoder'])
        nuplan_autoencoder.load_state_dict(state['nuplan_autoencoder'])
        l2d_autoencoder.eval()
        nuplan_autoencoder.eval()

        # L2D loader
        l2d_loader = DataLoader(
            l2d_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

        # NuPlan loader
        nuplan_loader = DataLoader(
            nuplan_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

        l2d_graph_embeddings = {}
        nuplan_graph_embeddings = {}

        from src.graph_encoding.autoencoder import batched_graph_embeddings

        with torch.no_grad():
            # L2D embeddings
            for data in tqdm(l2d_loader, desc="Extracting L2D embeddings"):
                data = l2d_quantizer.transform_inplace(data).to(device)
                z_dict, _, _ = l2d_autoencoder(data)
                graph_emb = batched_graph_embeddings(z_dict, data, l2d_dataset.get_metadata(), embed_dim_per_type=args.embed_dim)
                # Get the episode IDs from the episode_path
                print(data["window_meta"].episode_path) # Debug print
                episode_ids = [os.path.splitext(os.path.basename(p))[0].split('_')[0] for p in data["window_meta"].episode_path]

                for i, episode_id in enumerate(episode_ids):
                    l2d_graph_embeddings[episode_id] = graph_emb[i]

            # NuPlan embeddings
            for data in tqdm(nuplan_loader, desc="Extracting NuPlan embeddings"):
                data = nuplan_quantizer.transform_inplace(data).to(device)
                z_dict, _, _ = nuplan_autoencoder(data)
                graph_emb = batched_graph_embeddings(z_dict, data, nuplan_dataset.get_metadata(), embed_dim_per_type=args.embed_dim)
                episode_ids = [os.path.splitext(os.path.basename(p))[0].split('_')[0] for p in data["window_meta"].episode_path]
                for i, episode_id in enumerate(episode_ids):
                    nuplan_graph_embeddings[episode_id] = graph_emb[i]

        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(l2d_graph_embeddings, os.path.join(args.output_dir, "l2d_graph_embeddings.pt"))
        torch.save(nuplan_graph_embeddings, os.path.join(args.output_dir, "nuplan_graph_embeddings.pt"))
        print(f"Saved L2D and NuPlan graph embeddings to {args.output_dir}/")
    
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train, evaluate, and extract embeddings for graph autoencoder."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="null",
        help="Dataset name (used in paths under ./data/).",
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
        "--mode",
        type=str,
        default="all",
        help="Graph dataset mode.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Dimension of hidden layers.",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=32,
        help="Dimension of latent embeddings.",
    )
    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=1,
        help="Number of encoder layers.",
    )
    parser.add_argument(
        "--num_decoder_layers",
        type=int,
        default=1,
        help="Number of decoder layers.",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="Activation function.",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.1,
        help="Dropout rate.",
    )
    parser.add_argument(
        "--side_info_path",
        type=str,
        default=None,
        help="Path to side information file.",
    )
    parser.add_argument(
        "--risk_scores_path",
        type=str,
        default=None,
        help="Path to risk scores JSON file.",
    )
    parser.add_argument(
        "--node_features_to_exclude",
        nargs="+",
        type=str,
        default=None,
        help="List of node features to exclude.",
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
        "--kl_weight",
        type=float,
        default=0.1,
        help="Weight for the KL divergence loss.",
    )

    # Paths, seed, outputs
    parser.add_argument(
        "--best_model_path",
        type=str,
        default="./models/graph_encoder/best_model.pt",
        help="Path to save / load the best model checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/graph_embeddings/L2D",
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



    run_task(args)
