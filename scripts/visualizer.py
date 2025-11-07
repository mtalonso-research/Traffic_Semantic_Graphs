
import os
import argparse
import json
from src.visualizations import scene_visualizer, combined_graph_viewer

parser = argparse.ArgumentParser(description="Visualize data.")

# Arguments for both visualizers
parser.add_argument("--dataset_path", type=str, default='./data/graphical/L2D', help="Path to the dataset directory.")
parser.add_argument("--episode", type=int, default=0, help="Episode number to visualize.")

# Arguments for specific visualizers
parser.add_argument("--frame", type=lambda x: None if x.lower() == "none" else int(x), default=None, help="Frame number for scene visualizer.")
parser.add_argument("--graph_visualizer", action="store_true", help="Run the combined graph visualizer.")
parser.add_argument("--scene_visualizer", action="store_true", help="Run the original scene visualizer.")

args = parser.parse_args()

if __name__ == "__main__":
    if args.graph_visualizer:
        file_path = os.path.join(args.dataset_path, f"{args.episode}_graph.json")

        try:
            with open(file_path, 'r') as f:
                graph_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Graph file not found at {file_path}")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}")
        else:
            graphs_by_frame = {'0': graph_data}
            combined_graph_viewer(graphs_by_frame, args.episode)

    elif args.scene_visualizer:
        scene_visualizer(dataset_path=args.dataset_path,
                           episode=args.episode,
                           frame=args.frame)
    else:
        # Default to scene_visualizer if no flag is provided
        print("No visualizer specified. Defaulting to scene_visualizer.")
        scene_visualizer(dataset_path=args.dataset_path,
                           episode=args.episode,
                           frame=args.frame)
