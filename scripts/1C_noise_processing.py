
import os
import json
import argparse
import numpy as np
from tqdm import tqdm

def add_noise_to_graph(graph, noise_level):
    """
    Adds Gaussian noise to the features of ego and vehicle nodes in a graph.
    """
    for node_type in ['ego', 'vehicle']:
        if node_type in graph['nodes']:
            for node in graph['nodes'][node_type]:
                for feature in ['vx', 'vy', 'ax', 'ay', 'x', 'y', 'speed']:
                    if feature in node['features']:
                        noise = np.random.normal(0, noise_level)
                        node['features'][feature] += noise
    return graph

def process_graphs(data_dir, output_dir, noise_level):
    """
    Processes all graph JSONs in a directory, adds noise, and saves them to a new directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in tqdm(os.listdir(data_dir)):
        if filename.endswith('.json'):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r') as f:
                graph = json.load(f)

            noisy_graph = add_noise_to_graph(graph, noise_level)

            output_filepath = os.path.join(output_dir, filename)
            with open(output_filepath, 'w') as f:
                json.dump(noisy_graph, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add noise to graph JSONs.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the graph JSONs.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the noisy graphs.')
    parser.add_argument('--noise_level', type=float, required=True, help='Standard deviation of the Gaussian noise to add.')
    args = parser.parse_args()

    process_graphs(args.data_dir, args.output_dir, args.noise_level)

    print(f"Finished adding noise to graphs. Output saved to: {args.output_dir}")
