import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
REPO = Path(r"F:\NYCU\nuplan_clean\Traffic_Semantic_Graphs")
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from src.experiment_utils import ProjectionHead, infer_graph_emb_dim
from src.graph_encoding.autoencoder import HeteroGraphAutoencoder, QuantileFeatureQuantizer, batched_graph_embeddings
from src.graph_encoding.data_loaders import get_graph_dataset
from src.graph_encoding.risk_prediction import RiskPredictionHead


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(r"F:\NYCU\city_experiments\latency_report.csv"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--graphs", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = ckpt["args"]
    view = REPO / "data/NuPlan/city_views/singapore_to_boston"
    train_ds = get_graph_dataset(
        root_dir=view / "training_data/noisy_0/graphs",
        mode=saved.get("mode", "all"),
        side_information_path=None,
        risk_scores_path=view / "training_data/noisy_0/risk_scores.json",
    )
    eval_ds = get_graph_dataset(
        root_dir=view / "evaluation_data/noisy_0/graphs",
        mode=saved.get("mode", "all"),
        side_information_path=None,
        risk_scores_path=view / "evaluation_data/noisy_0/risk_scores_true.json",
    )
    metadata = train_ds.get_metadata()
    quantizer = QuantileFeatureQuantizer(bins=saved.get("quant_bins", 32), node_types=metadata[0])
    quantizer.fit(train_ds)
    encoder = HeteroGraphAutoencoder(
        metadata=metadata,
        hidden_dim=saved.get("hidden_dim", 64),
        embed_dim=saved.get("embed_dim", 64),
        quantizer_spec=quantizer.spec(),
        feat_emb_dim=16,
        num_encoder_layers=saved.get("num_encoder_layers", 1),
        num_decoder_layers=saved.get("num_decoder_layers", 1),
        activation=saved.get("activation", "relu"),
        dropout_rate=saved.get("dropout_rate", 0.1),
        side_info_dim=0,
    ).to(device)
    infer_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0)
    graph_dim = infer_graph_emb_dim(encoder, quantizer, infer_loader, metadata, embed_dim_per_type=saved.get("embed_dim", 64))
    projection = ProjectionHead(
        dim=graph_dim,
        hidden_dim=saved.get("proj_hidden_dim") or None,
        dropout=saved.get("proj_dropout", 0.05),
        activation=saved.get("proj_activation", "relu"),
        residual=bool(saved.get("proj_residual", True)),
    ).to(device)
    risk_head = RiskPredictionHead(
        input_dim=graph_dim,
        hidden_dim=saved.get("risk_hidden_dim", 64),
        output_dim=saved.get("num_classes", 4),
        mode="classification",
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_noisy_state_dict"])
    projection.load_state_dict(ckpt["proj_noisy_state_dict"])
    risk_head.load_state_dict(ckpt["risk_head_state_dict"])
    encoder.eval(); projection.eval(); risk_head.eval()
    loader = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=0)

    def forward(batch):
        batch = quantizer.transform_inplace(batch).to(device)
        z_dict, _, _ = encoder(batch)
        embedding = batched_graph_embeddings(z_dict, batch, metadata, embed_dim_per_type=saved.get("embed_dim", 64))
        projected = projection(embedding)
        if saved.get("proj_l2_normalize", False):
            projected = F.normalize(projected, dim=-1)
        return risk_head(projected)

    with torch.no_grad():
        warmed = 0
        while warmed < args.warmup:
            for batch in loader:
                forward(batch)
                warmed += 1
                if warmed >= args.warmup:
                    break
        if device.type == "cuda": torch.cuda.synchronize()
        timings = []
        for index, batch in enumerate(loader):
            if index >= args.graphs: break
            if device.type == "cuda": torch.cuda.synchronize()
            start = time.perf_counter()
            forward(batch)
            if device.type == "cuda": torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)

    row = {
        "checkpoint": str(args.checkpoint),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "batch_size": 1,
        "warmup_graphs": args.warmup,
        "graphs_tested": len(timings),
        "mean_ms_per_graph": statistics.mean(timings),
        "median_ms_per_graph": statistics.median(timings),
        "p95_ms_per_graph": float(np.percentile(timings, 95)),
    }
    with args.output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader(); writer.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
