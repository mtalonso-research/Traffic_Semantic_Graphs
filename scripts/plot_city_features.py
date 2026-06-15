import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FEATURES = [
    "duration",
    "frames",
    "ego_speed_mean",
    "ego_speed_std",
    "ego_accel_mean",
    "ego_accel_max",
    "ego_heading_change",
    "vehicle_count",
    "vehicle_dist_mean",
    "vehicle_dist_min",
    "vehicle_speed_mean",
    "pedestrian_count",
    "pedestrian_dist_mean",
    "object_count",
    "object_dist_mean",
    "edge_count_total",
    "edge_count_ego_to_vehicle",
    "edge_count_ego_to_pedestrian",
    "edge_count_ego_to_object",
    "env_time_mean",
    "env_daylight_frac",
    "env_precipitation_mean",
]


def normalize_city(city: str) -> str:
    return city.strip().split("_")[-1].lower()


def numeric_sort_key(path: Path):
    token = path.name.split("_")[0]
    return (0, int(token)) if token.isdigit() else (1, path.name)


def city_from_graph(data: Dict) -> str:
    return normalize_city(data.get("metadata", {}).get("city", ""))


def graph_paths_for_city(data_root: Path, split: str, city: str) -> List[Path]:
    city_root = data_root / split / f"clean_{city}" / "graphs"
    if city_root.is_dir():
        return sorted(city_root.glob("*.json"), key=numeric_sort_key)

    clean_root = data_root / split / "clean" / "graphs"
    if not clean_root.is_dir():
        raise FileNotFoundError(f"No graph directory found for {city}: {city_root} or {clean_root}")

    paths = []
    for path in sorted(clean_root.glob("*.json"), key=numeric_sort_key):
        with path.open("r") as f:
            data = json.load(f)
        if city_from_graph(data) == city:
            paths.append(path)
    return paths


def sample_paths(paths: Sequence[Path], max_count: int, seed: int) -> List[Path]:
    paths = list(paths)
    if max_count <= 0 or len(paths) <= max_count:
        return paths
    rng = random.Random(seed)
    return sorted(rng.sample(paths, max_count), key=numeric_sort_key)


def numeric_values(nodes: Iterable[Dict], key: str) -> List[float]:
    values = []
    for node in nodes:
        value = node.get("features", {}).get(key)
        if isinstance(value, bool):
            values.append(float(value))
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def mean_or_zero(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def std_or_zero(values: Sequence[float]) -> float:
    return float(np.std(values)) if values else 0.0


def min_or_zero(values: Sequence[float]) -> float:
    return float(np.min(values)) if values else 0.0


def max_or_zero(values: Sequence[float]) -> float:
    return float(np.max(values)) if values else 0.0


def extract_scene_features(path: Path) -> Dict[str, float]:
    with path.open("r") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    nodes = data.get("nodes", {})
    edges = data.get("edges", {})

    ego = nodes.get("ego", [])
    vehicles = nodes.get("vehicle", [])
    pedestrians = nodes.get("pedestrian", [])
    objects = nodes.get("object", [])
    environment = nodes.get("environment", [])

    ego_speed = numeric_values(ego, "speed")
    ego_ax = numeric_values(ego, "ax")
    ego_ay = numeric_values(ego, "ay")
    ego_accel = [
        math.sqrt((ego_ax[i] ** 2) + (ego_ay[i] ** 2))
        for i in range(min(len(ego_ax), len(ego_ay)))
    ]
    ego_heading = numeric_values(ego, "heading")
    ego_heading_change = abs(ego_heading[-1] - ego_heading[0]) if len(ego_heading) >= 2 else 0.0

    vehicle_dist = numeric_values(vehicles, "dist_to_ego")
    vehicle_speed = numeric_values(vehicles, "speed")
    pedestrian_dist = numeric_values(pedestrians, "dist_to_ego")
    pedestrian_speed = numeric_values(pedestrians, "speed")
    object_dist = numeric_values(objects, "dist_to_ego")

    env_time = numeric_values(environment, "time")
    env_daylight = numeric_values(environment, "daylight")
    env_weekend = numeric_values(environment, "weekend")
    env_precip = numeric_values(environment, "precipitation")
    env_conditions = numeric_values(environment, "conditions")

    feature_row = {
        "duration": float(metadata.get("t_end", 0.0)) - float(metadata.get("t_start", 0.0)),
        "frames": float(metadata.get("frames", len(ego))),
        "ego_speed_mean": mean_or_zero(ego_speed),
        "ego_speed_std": std_or_zero(ego_speed),
        "ego_speed_max": max_or_zero(ego_speed),
        "ego_accel_mean": mean_or_zero(ego_accel),
        "ego_accel_max": max_or_zero(ego_accel),
        "ego_heading_change": ego_heading_change,
        "vehicle_count": float(len(vehicles)),
        "vehicle_dist_mean": mean_or_zero(vehicle_dist),
        "vehicle_dist_min": min_or_zero(vehicle_dist),
        "vehicle_speed_mean": mean_or_zero(vehicle_speed),
        "vehicle_speed_max": max_or_zero(vehicle_speed),
        "pedestrian_count": float(len(pedestrians)),
        "pedestrian_dist_mean": mean_or_zero(pedestrian_dist),
        "pedestrian_dist_min": min_or_zero(pedestrian_dist),
        "pedestrian_speed_mean": mean_or_zero(pedestrian_speed),
        "object_count": float(len(objects)),
        "object_dist_mean": mean_or_zero(object_dist),
        "object_dist_min": min_or_zero(object_dist),
        "edge_count_total": float(sum(len(items) for items in edges.values())),
        "edge_count_ego_to_vehicle": float(len(edges.get("ego_to_vehicle", []))),
        "edge_count_ego_to_pedestrian": float(len(edges.get("ego_to_pedestrian", []))),
        "edge_count_ego_to_object": float(len(edges.get("ego_to_object", []))),
        "edge_count_ego_to_environment": float(len(edges.get("ego_to_environment", []))),
        "env_time_mean": mean_or_zero(env_time),
        "env_daylight_frac": mean_or_zero(env_daylight),
        "env_weekend_frac": mean_or_zero(env_weekend),
        "env_precipitation_mean": mean_or_zero(env_precip),
        "env_conditions_mean": mean_or_zero(env_conditions),
    }
    return feature_row


def load_city_rows(data_root: Path, split: str, city: str, max_per_city: int, seed: int) -> List[Dict[str, float]]:
    paths = graph_paths_for_city(data_root, split, city)
    sampled = sample_paths(paths, max_count=max_per_city, seed=seed)
    rows = []
    for path in sampled:
        row = extract_scene_features(path)
        row["city"] = city
        row["episode"] = path.name.split("_")[0]
        row["path"] = str(path)
        rows.append(row)
    print(f"[plot-city] {city}: using {len(rows)} of {len(paths)} {split} graphs")
    return rows


def project_2d(features: np.ndarray, method: str, seed: int) -> Tuple[np.ndarray, str, List[float]]:
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    scaled = (features - mean) / std

    if method == "pca":
        _u, s, vt = np.linalg.svd(scaled, full_matrices=False)
        coords = scaled @ vt[:2].T
        variance = (s ** 2) / max(len(features) - 1, 1)
        total_variance = float(np.sum(variance))
        explained = (
            [float(v / total_variance) for v in variance[:2]]
            if total_variance > 0
            else [0.0, 0.0]
        )
        subtitle = f"PCA explained variance: PC1={explained[0]:.2%}, PC2={explained[1]:.2%}"
        return coords, subtitle, explained

    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise SystemExit("--method tsne requires scikit-learn. Use --method pca or install sklearn.") from exc

    perplexity = min(30, max(5, (len(features) - 1) // 3))
    model = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    coords = model.fit_transform(scaled)
    return coords, f"t-SNE projection, perplexity={perplexity}", []


def save_feature_csv(path: Path, rows: List[Dict], feature_names: Sequence[str], coords: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["city", "episode", "x_2d", "y_2d", *feature_names, "path"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, coord in zip(rows, coords):
            out = {key: row.get(key, "") for key in fieldnames}
            out["x_2d"] = float(coord[0])
            out["y_2d"] = float(coord[1])
            writer.writerow(out)


def plot_projection(
    rows: List[Dict],
    coords: np.ndarray,
    source_city: str,
    target_city: str,
    subtitle: str,
    output_path: Path,
) -> None:
    colors = {target_city: "#1f77b4", source_city: "#d62728"}
    labels = {target_city: f"{target_city} (blue)", source_city: f"{source_city} (red)"}

    fig, ax = plt.subplots(figsize=(10, 8))
    for city in [target_city, source_city]:
        mask = np.array([row["city"] == city for row in rows])
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            alpha=0.55,
            c=colors[city],
            label=labels[city],
            edgecolors="none",
        )

    ax.set_title(f"Clean NuPlan Scene Features: {target_city} vs {source_city}", fontsize=14)
    ax.text(
        0.01,
        0.99,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(alpha=0.18)
    ax.legend(frameon=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_feature_names(raw: str) -> List[str]:
    if raw.strip().lower() == "default":
        return DEFAULT_FEATURES
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 2D clustering of selected clean graph features for two NuPlan cities."
    )
    parser.add_argument("--data_root", type=Path, default=Path("data/NuPlan"))
    parser.add_argument("--split", choices=["training_data", "evaluation_data"], default="training_data")
    parser.add_argument("--source_city", type=str, default="singapore", help="Plotted in red.")
    parser.add_argument("--target_city", type=str, default="boston", help="Plotted in blue.")
    parser.add_argument("--max_per_city", type=int, default=1500, help="<=0 uses every available graph.")
    parser.add_argument("--seed", type=int, default=228)
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca")
    parser.add_argument(
        "--features",
        type=str,
        default="default",
        help="Comma-separated feature names, or 'default'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--save_csv",
        type=Path,
        default=Path("experiment_results/city_feature_projection.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_city = normalize_city(args.source_city)
    target_city = normalize_city(args.target_city)
    feature_names = parse_feature_names(args.features)
    output_path = args.output or Path(
        f"figures/city_feature_clusters_{source_city}_{target_city}_{args.method}.png"
    )

    rows = []
    rows.extend(load_city_rows(args.data_root, args.split, target_city, args.max_per_city, args.seed))
    rows.extend(load_city_rows(args.data_root, args.split, source_city, args.max_per_city, args.seed + 1))
    if len(rows) < 3:
        raise SystemExit("Need at least three graphs total to make a 2D projection.")

    missing = [name for name in feature_names if name not in rows[0]]
    if missing:
        raise SystemExit(f"Unknown feature(s): {missing}. Use --features default or inspect the script list.")

    feature_matrix = np.array([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float64)
    coords, subtitle, explained = project_2d(feature_matrix, method=args.method, seed=args.seed)
    plot_projection(rows, coords, source_city, target_city, subtitle, output_path)
    save_feature_csv(args.save_csv, rows, feature_names, coords)

    print(f"[plot-city] features: {', '.join(feature_names)}")
    if explained:
        print(f"[plot-city] PCA explained variance: {explained[0]:.4f}, {explained[1]:.4f}")
    print(f"[plot-city] saved plot: {output_path}")
    print(f"[plot-city] saved projected feature table: {args.save_csv}")


if __name__ == "__main__":
    main()
