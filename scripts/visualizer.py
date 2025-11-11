import os
import argparse
import json
from src.visualizations import scene_visualizer, combined_graph_viewer, plot_feature_histogram

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize data.")

    parser.add_argument("--dataset_path", type=str, default='./data/graphical/L2D', help="Path to the dataset directory.")
    parser.add_argument("--episode", type=int, default=0, help="Episode number to visualize.")

    parser.add_argument("--frame", type=lambda x: None if x.lower() == "none" else int(x), default=None, help="Frame number for scene visualizer.")
    parser.add_argument("--graph_visualizer", action="store_true", help="Run the combined graph visualizer.")
    parser.add_argument("--scene_visualizer", action="store_true", help="Run the original scene visualizer.")
    parser.add_argument("--histogram", type=str, default=None, help="Plot a histogram of a feature (e.g., 'ego-vx' or 'ego-all').")

    args = parser.parse_args()

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
            combined_graph_viewer(graphs_by_frame, args.episode, args.dataset_path)

    elif args.scene_visualizer:
        scene_visualizer(dataset_path=args.dataset_path,
                           episode=args.episode,
                           frame=args.frame)
    
    elif args.histogram:
        try:
            node_type, feature_name = args.histogram.split('-')
            plot_feature_histogram(data_directory=args.dataset_path,
                                   node_type=node_type,
                                   feature_name=feature_name)
        except ValueError:
            print("Error: Invalid format for --histogram argument. Please use 'node_type-feature_name' (e.g., 'ego-vx').")

    else:
        print("No visualizer specified.")
