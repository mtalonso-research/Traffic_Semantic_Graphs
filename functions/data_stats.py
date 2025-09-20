import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_tag_distributions(directory_path: str):
    """
    Crawl all JSON files in a directory, extract tag frequencies, 
    and plot distributions for action, traffic control, road feature, and environment tags.
    """

    # Counters for each tag type
    action_counts = Counter()
    traffic_counts = Counter()
    road_counts = Counter()
    env_counts = Counter()

    # Normalize keys (singular/plural variants)
    key_map = {
        "action_tags": "action",
        "action_tag": "action",
        "traffic_control_tags": "traffic",
        "traffic_control_tag": "traffic",
        "road_feature_tags": "road",
        "road_feature_tag": "road",
        "environment_tags": "env",
        "environment_tag": "env"
    }

    # Crawl directory
    for filename in os.listdir(directory_path):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(directory_path, filename)
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {filename} (error: {e})")
            continue

        # Process tags
        for key, category in key_map.items():
            if key in data:
                tags = data[key]
                if isinstance(tags, str):  # normalize single string
                    tags = [tags]
                if category == "action":
                    action_counts.update(tags)
                elif category == "traffic":
                    traffic_counts.update(tags)
                elif category == "road":
                    road_counts.update(tags)
                elif category == "env":
                    env_counts.update(tags)

    # Helper for plotting
    def plot_counter(counter, title):
        if not counter:
            print(f"No data for {title}")
            return
        tags, counts = zip(*counter.most_common())
        plt.figure(figsize=(10, 6))
        plt.bar(tags, counts)
        plt.title(f"Distribution of {title}")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    # Plot all four
    plot_counter(action_counts, "Action Tags")
    plot_counter(traffic_counts, "Traffic Control Tags")
    plot_counter(road_counts, "Road Feature Tags")
    plot_counter(env_counts, "Environment Tags")



def analyze_feature_distributions(dataset_dir: str, dataset_name: str):
    """
    Loads all JSON graphs in dataset_dir, extracts features, and plots distributions.

    Args:
        dataset_dir: Path to processed_graphs/<dataset> directory.
        dataset_name: Label for the dataset (used in plot titles).
    """

    # --- Load graphs ---
    rows = []
    for p in Path(dataset_dir).glob("*.json"):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for node_type, nodes in data.get("nodes", {}).items():
            for n in nodes:
                feats = n.get("features", {})
                for k, v in feats.items():
                    rows.append({
                        "node_type": node_type,
                        "feature": k,
                        "value": v
                    })
    df = pd.DataFrame(rows)

    if df.empty:
        print(f"[{dataset_name}] No data found in {dataset_dir}")
        return

    # --- Numeric features ---
    numeric_feats = ["speed", "vx", "vy", "vz", "distance_to_ego"]

    for feat in numeric_feats:
        vals = df[df["feature"] == feat]["value"].dropna()
        if vals.empty:
            continue
        plt.figure()
        plt.hist(vals, bins=50, alpha=0.7, color="C0", density=True)
        plt.title(f"{dataset_name} - Distribution of {feat}")
        plt.xlabel(feat)
        plt.ylabel("Density")
        plt.show()

    # --- Categorical features ---
    categorical_feats = ["month", "day_of_week"]

    for feat in categorical_feats:
        subset = df[df["feature"] == feat]
        if subset.empty:
            continue
        counts = subset.groupby("value").size()
        counts.plot(kind="bar", figsize=(8,4), color="C1")
        plt.title(f"{dataset_name} - Distribution of {feat}")
        plt.ylabel("Count")
        plt.show()

    # --- Time of day: bin into hours ---
    time_subset = df[df["feature"] == "time_of_day"].copy()
    if not time_subset.empty:
        time_subset["hour"] = pd.to_datetime(
            time_subset["value"], format="%H:%M:%S", errors="coerce"
        ).dt.hour

        vals = time_subset["hour"].dropna()
        if not vals.empty:
            plt.figure()
            plt.hist(vals, bins=24, alpha=0.7, color="C2", density=True)
            plt.title(f"{dataset_name} - Distribution of time_of_day (hours)")
            plt.xlabel("Hour of day")
            plt.ylabel("Density")
            plt.show()
