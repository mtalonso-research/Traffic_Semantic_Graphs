import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import os
import json
from tqdm import tqdm
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_encoding.risk_prediction import RiskPredictionHead, EmbeddingRiskDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_task(args):
    if "best_model.pt" in args.best_model_path:
        side_info_str = "_with_side_info" if args.side_info_4b else ""
        args.best_model_path = f"./models/risk_predictor_from_embeddings/4C{side_info_str}_best_model.pt"

    # Load embeddings
    nuplan_embeddings = torch.load(os.path.join(args.embedding_dir, args.nuplan_embedding_file))
    l2d_embeddings = torch.load(os.path.join(args.embedding_dir, args.l2d_embedding_file))

    # Load risk scores
    with open(args.nuplan_risk_scores_path, 'r') as f:
        nuplan_risk_scores = json.load(f)
    with open(args.l2d_risk_scores_path, 'r') as f:
        l2d_risk_scores = json.load(f)

    # Create datasets
    nuplan_dataset = EmbeddingRiskDataset(nuplan_embeddings, nuplan_risk_scores)
    l2d_val_dataset = EmbeddingRiskDataset(l2d_embeddings, l2d_risk_scores)

    # Split NuPlan dataset
    nuplan_train_size = int((1 - args.val_fraction) * len(nuplan_dataset))
    nuplan_val_size = len(nuplan_dataset) - nuplan_train_size
    nuplan_train_dataset, nuplan_val_dataset = random_split(nuplan_dataset, [nuplan_train_size, nuplan_val_size])

    # Create dataloaders
    train_loader = DataLoader(nuplan_train_dataset, batch_size=args.batch_size, shuffle=True)
    nuplan_val_loader = DataLoader(nuplan_val_dataset, batch_size=args.batch_size, shuffle=False)
    l2d_val_loader = DataLoader(l2d_val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    embedding_dim = next(iter(nuplan_embeddings.values())).shape[0]
    output_dim = 1 if args.prediction_mode == "regression" else 4
    prediction_head = RiskPredictionHead(
        input_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        mode=args.prediction_mode
    ).to(device)

    # Loss and optimizer
    if args.prediction_mode == "regression":
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(prediction_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float('inf')

    if args.train:
        print("Training risk prediction head on NuPlan embeddings...")
        for epoch in range(1, args.num_epochs + 1):
            prediction_head.train()
            total_loss = 0
            for embeddings, risk_scores in tqdm(train_loader, desc=f"Epoch {epoch}"):
                embeddings = embeddings.to(device)
                risk_scores = risk_scores.to(device)

                optimizer.zero_grad()
                pred = prediction_head(embeddings)
                loss = loss_fn(pred, risk_scores.view(-1, 1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch:02d} | Train Loss: {avg_loss:.4f}")

            # Validation loop on NuPlan
            prediction_head.eval()
            total_val_loss = 0
            with torch.no_grad():
                for embeddings, risk_scores in tqdm(nuplan_val_loader, desc="NuPlan Validation"):
                    embeddings = embeddings.to(device)
                    risk_scores = risk_scores.to(device)
                    pred = prediction_head(embeddings)
                    loss = loss_fn(pred, risk_scores.view(-1, 1))
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(nuplan_val_loader)
            print(f"NuPlan Val Loss: {avg_val_loss:.4f}")

            # Save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs(os.path.dirname(args.best_model_path), exist_ok=True)
                torch.save(prediction_head.state_dict(), args.best_model_path)
                print(f"  -> New best model saved to {args.best_model_path}")

    if args.evaluate:
        print("Evaluating model...")
        prediction_head.load_state_dict(torch.load(args.best_model_path))
        prediction_head.eval()

        # Evaluation on NuPlan
        total_nuplan_loss = 0
        with torch.no_grad():
            for embeddings, risk_scores in tqdm(nuplan_val_loader, desc="NuPlan Evaluation"):
                embeddings = embeddings.to(device)
                risk_scores = risk_scores.to(device)
                pred = prediction_head(embeddings)
                loss = loss_fn(pred, risk_scores.view(-1, 1))
                total_nuplan_loss += loss.item()
        
        avg_nuplan_loss = total_nuplan_loss / len(nuplan_val_loader)
        print(f"Final NuPlan Validation Loss: {avg_nuplan_loss:.4f}")

        # Evaluation on L2D
        total_l2d_loss = 0
        with torch.no_grad():
            for embeddings, risk_scores in tqdm(l2d_val_loader, desc="L2D Evaluation"):
                embeddings = embeddings.to(device)
                risk_scores = risk_scores.to(device)
                pred = prediction_head(embeddings)
                loss = loss_fn(pred, risk_scores.view(-1, 1))
                total_l2d_loss += loss.item()

        avg_l2d_loss = total_l2d_loss / len(l2d_val_loader)
        print(f"Final L2D Validation Loss: {avg_l2d_loss:.4f}")

        if args.save_annotations:
            print("Saving evaluation results...")
            results = {}
            if os.path.exists("evaluation_results.json"):
                with open("evaluation_results.json", 'r') as f:
                    results = json.load(f)
            
            if args.side_info_4b:
                results["4C"]["L2D_with_side_info"] = avg_l2d_loss
                results["4C"]["NuPlan_with_side_info_4b"] = avg_nuplan_loss
            else:
                results["4C"]["L2D_without_side_info"] = avg_l2d_loss
                results["4C"]["NuPlan_without_side_info_4b"] = avg_nuplan_loss

            with open("evaluation_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            print("Evaluation results saved to evaluation_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a risk prediction model from graph embeddings.")

    # Data args
    parser.add_argument("--embedding_dir", type=str, default="./data/graph_embeddings/", help="Directory where graph embeddings are stored.")
    parser.add_argument("--nuplan_embedding_file", type=str, default="nuplan_graph_embeddings.pt", help="NuPlan embedding file name.")
    parser.add_argument("--l2d_embedding_file", type=str, default="l2d_graph_embeddings.pt", help="L2D embedding file name.")
    parser.add_argument("--nuplan_risk_scores_path", type=str, default="./data/graph_dataset/risk_scores_NuPlan.json", help="Path to NuPlan risk scores JSON file.")
    parser.add_argument("--l2d_risk_scores_path", type=str, default="data/validation_dataset/L2D/risk_outputs/risk_analysis_L2D_val.json", help="Path to L2D validation risk scores JSON file.")
    
    # Model / training hyperparams
    parser.add_argument("--hidden_dim", type=int, default=64, help="Dimension of hidden layers.")
    parser.add_argument("--prediction_mode", type=str, default="regression", choices=["regression", "classification"], help="Risk prediction mode.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--val_fraction", type=float, default=0.2, help="Fraction of data for validation.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay.")

    # Task args
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model.")
    parser.add_argument("--save_annotations", action="store_true", help="Save evaluation losses to a file.")
    parser.add_argument("--side_info_4b", action="store_true", help="Flag to indicate that the embeddings from 4B were trained with side information.")

    # Paths, seed, outputs
    parser.add_argument("--best_model_path", type=str, default="./models/risk_predictor_from_embeddings/best_model.pt", help="Path to save/load the best model checkpoint.")
    
    args = parser.parse_args()
    run_task(args)
