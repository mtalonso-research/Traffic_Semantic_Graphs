import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import os
import numpy as np
from tqdm import tqdm
import sys
import wandb
from torch.utils.data import random_split
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_encoding.data_loaders import get_graph_dataset
from src.graph_encoding.autoencoder import HeteroGraphAutoencoder, batched_graph_embeddings, QuantileFeatureQuantizer
from src.graph_encoding.risk_prediction import RiskPredictionHead

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def risk_to_class(risk_scores):
    """Converts risk scores to classes."""
    # risk_scores is a tensor of shape (batch_size, 1, 1)
    risk_scores = risk_scores.squeeze()
    classes = torch.zeros_like(risk_scores, dtype=torch.long)
    classes[risk_scores > 0.25] = 1
    classes[risk_scores > 0.5] = 2
    classes[risk_scores > 0.75] = 3
    return classes

def run_task(args):

    if args.sweep:
        wandb.init(config=args)
        args = wandb.config
    
    if not args.l2d and not args.nup:
        print("Please specify a dataset to process using --l2d or --nup.")
        return

    dataset_name = "L2D" if args.l2d else "NuPlan"

    if "best_model.pt" in args.best_model_path:
        side_info_str = "_with_side_info" if args.with_side_information else ""
        args.best_model_path = f"./models/risk_predictor/4A_{dataset_name}{side_info_str}_best_model.pt"
    
    root_dir = os.path.join(args.input_directory, dataset_name)
    risk_scores_path = os.path.join(args.input_directory, f"risk_scores_{dataset_name}.json")
    side_information_path = None
    if args.with_side_information and args.l2d:
        side_information_path = os.path.join(args.input_directory, "L2D_frame_embs")

    print('Loading in Dataset ...')
    dataset = get_graph_dataset(
        root_dir=root_dir,
        mode=args.mode,
        side_information_path=side_information_path,
        risk_scores_path=risk_scores_path
    )

    print('Quantizing Dataset ...')
    quantizer = QuantileFeatureQuantizer(bins=32, node_types=dataset.get_metadata()[0])
    quantizer.fit(dataset)

    print('Spliting Dataset ...')
    val_size = int(args.val_fraction * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print('Initializing Models ...')
    side_info_dim = dataset.side_info_dim if args.with_side_information else 0
    encoder = HeteroGraphAutoencoder(
        metadata=dataset.get_metadata(),
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        quantizer_spec=quantizer.spec(),
        feat_emb_dim=16,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=1, # Not used
        activation=args.activation,
        dropout_rate=args.dropout_rate,
        side_info_dim=side_info_dim,
    ).to(device)
    
    num_node_types = len(dataset.get_metadata()[0])
    graph_embed_dim = num_node_types * args.embed_dim
    output_dim = 1 if args.prediction_mode == "regression" else 4
    prediction_head = RiskPredictionHead(
        input_dim=graph_embed_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        mode=args.prediction_mode
    ).to(device)

    # Loss and optimizer
    if args.prediction_mode == "regression":
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(prediction_head.parameters()), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )

    best_val_loss = float('inf')

    if args.train:
        print(f"Training risk prediction model on {dataset_name}...")
        for epoch in range(1, args.num_epochs + 1):
            encoder.train()
            prediction_head.train()
            total_loss = 0
            for data in tqdm(train_loader, desc=f"Epoch {epoch}"):
                data = data.to(device)
                data = quantizer.transform_inplace(data) # Apply quantizer
                optimizer.zero_grad()

                z_dict, _, _ = encoder(data)
                graph_emb = batched_graph_embeddings(z_dict, data, dataset.get_metadata(), embed_dim_per_type=args.embed_dim)

                pred = prediction_head(graph_emb)
                
                if args.prediction_mode == "regression":
                    target = data.y.view(-1, 1)
                else:
                    target = risk_to_class(data.y)
                
                loss = loss_fn(pred, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            
            avg_loss = total_loss / len(train_loader)
            if args.sweep:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": avg_loss
                })
            print(f"Epoch {epoch:02d} | Train Loss: {avg_loss:.4f}", end=" | ")

            # Validation loop
            encoder.eval()
            prediction_head.eval()
            total_val_loss = 0
            with torch.no_grad():
                for data in tqdm(val_loader, desc=f"Validation"):
                    data = data.to(device)
                    data = quantizer.transform_inplace(data) # Apply quantizer
                    
                    z_dict, _, _ = encoder(data)
                    graph_emb = batched_graph_embeddings(z_dict, data, dataset.get_metadata(), embed_dim_per_type=args.embed_dim)

                    pred = prediction_head(graph_emb)

                    if args.prediction_mode == "regression":
                        target = data.y.view(-1, 1)
                    else:
                        target = risk_to_class(data.y)
                    
                    loss = loss_fn(pred, target)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            if args.sweep:
                wandb.log({
                    "epoch": epoch,
                    "val_loss": avg_val_loss
                })
            print(f"Val Loss: {avg_val_loss:.4f}")

            # Save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs(os.path.dirname(args.best_model_path), exist_ok=True)
                torch.save(prediction_head.state_dict(), args.best_model_path)
                print(f"  -> New best model saved to {args.best_model_path}")

    if args.evaluate:
        print(f"Evaluating model on {dataset_name}...")
        
        # Load the validation data loader
        if args.l2d:
            val_dataset = get_graph_dataset(
                root_dir=args.val_dir,
                mode=args.mode,
                side_information_path=side_information_path,
                risk_scores_path=args.val_risk_scores_path
            )
            eval_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        else:
            eval_loader = val_loader

        prediction_head.load_state_dict(torch.load(args.best_model_path))
        encoder.eval()
        prediction_head.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data in tqdm(eval_loader, desc=f"Evaluation"):
                data = data.to(device)
                data = quantizer.transform_inplace(data)
                
                z_dict, _, _ = encoder(data)
                graph_emb = batched_graph_embeddings(z_dict, data, dataset.get_metadata(), embed_dim_per_type=args.embed_dim)

                pred = prediction_head(graph_emb)

                if args.prediction_mode == "regression":
                    target = data.y.view(-1, 1)
                else:
                    target = risk_to_class(data.y)
                
                loss = loss_fn(pred, target)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(eval_loader)
        print(f"Final Validation Loss: {avg_val_loss:.4f}")

        if args.save_annotations:
            print("Saving evaluation results...")
            results = {}
            if os.path.exists("evaluation_results.json"):
                with open("evaluation_results.json", 'r') as f:
                    results = json.load(f)
            
            if args.l2d:
                if args.with_side_information:
                    results["4A"]["L2D_with_side_info"] = avg_val_loss
                else:
                    results["4A"]["L2D_without_side_info"] = avg_val_loss
            elif args.nup:
                results["4A"]["NuPlan"] = avg_val_loss

            with open("evaluation_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            print("Evaluation results saved to evaluation_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a risk prediction model.")

    # Data args
    parser.add_argument("--input_directory", type=str, default="data/graph_dataset/", help="Input directory for graph data.")
    parser.add_argument("--val_dir", type=str, default="data/validation_dataset/L2D/graphical_final", help="Directory for validation graphs.")
    parser.add_argument("--val_risk_scores_path", type=str, default="data/validation_dataset/L2D/risk_outputs/risk_analysis_L2D_val.json", help="Path to validation risk scores.")
    parser.add_argument("--l2d", action="store_true", help="Process L2D dataset.")
    parser.add_argument("--nup", action="store_true", help="Process NuPlan dataset.")
    parser.add_argument("--with_side_information", action="store_true", help="Load side information for L2D dataset.")

    # Model / training hyperparams
    parser.add_argument("--mode", type=str, default="all", help="Graph dataset mode.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Dimension of hidden layers.")
    parser.add_argument("--embed_dim", type=int, default=32, help="Dimension of latent embeddings.")
    parser.add_argument("--num_encoder_layers", type=int, default=1, help="Number of encoder layers.")
    parser.add_argument("--activation", type=str, default="relu", help="Activation function.")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument("--val_fraction", type=float, default=0.2, help="Fraction of data for validation.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay.")
    
    # Task args
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model.")
    parser.add_argument("--save_annotations", action="store_true", help="Save evaluation losses to a file.")
    parser.add_argument("--prediction_mode", type=str, default="regression", choices=["regression", "classification"], help="Risk prediction mode.")
    parser.add_argument("--sweep", action="store_true", help="Run a wandb sweep.")

    # Paths, seed, outputs
    parser.add_argument("--best_model_path", type=str, default="./models/risk_predictor/best_model.pt", help="Path to save/load the best model checkpoint.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()
    run_task(args)