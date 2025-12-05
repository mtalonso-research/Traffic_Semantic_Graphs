import argparse
import os
import pandas as pd
import json
from src.risk_analysis.risk_analysis import RiskAnalysis
from src.utils import extract_frames
import numpy as np
from tqdm import tqdm

def run_risk_analysis_on_all_episodes(input_directory, output_directory, output_filename):
    """
    Runs the risk analysis on all episodes in a given directory and saves the results to a JSON file.
    """
    print("========== Starting Risk Analysis on all episodes ==========")
    risk_analyzer = RiskAnalysis()
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    results = {}
    
    episode_files = [f for f in os.listdir(input_directory) if f.endswith('_graph.json')]

    for episode_file in tqdm(episode_files):
        episode_num = int(episode_file.split('_')[0])
        file_path = os.path.join(input_directory, episode_file)
        
        with open(file_path, "r") as f:
            graph = json.load(f)

        frames = extract_frames(graph)
        if not frames:
            continue

        risks = []
        for i, frame in enumerate(frames):
            try:
                R, _, _, _ = risk_analyzer.analyze_frame(frame)
                risks.append(R)
            except Exception as e:
                continue

        if not risks:
            continue

        max_risk = float(np.max(risks))
        results[episode_num] = max_risk

    if not output_filename.endswith(".json"):
        output_filename += ".json"
    output_path = os.path.join(output_directory, output_filename)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Risk analysis complete. Results saved to: {output_path}")
    print("=======================================================")


def run_risk_analysis(input_directory, output_directory, num_episodes):
    """
    Runs the risk analysis on the graph data and saves the results to a CSV file.
    """
    print("========== Starting Risk Analysis ==========")
    risk_analyzer = RiskAnalysis()
    print(f"Collecting risk data from: {input_directory}")
    risk_df = risk_analyzer.collect_risk_data(input_directory, num_episodes)
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    output_path = os.path.join(output_directory, "risk_data.csv")
    risk_df.to_csv(output_path, index=False)
    
    print(f"Risk analysis complete. Results saved to: {output_path}")
    print("==========================================")

def extract_risk_episodes(csv_path, threshold, high_risk=True):
    """
    Prints episode numbers with risk above or below a certain threshold.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file at {csv_path} was not found.")
        return

    if high_risk:
        filtered_eps = df[df['max_risk'] > threshold]
        print(f"Episodes with max_risk > {threshold}:")
    else:
        filtered_eps = df[df['max_risk'] < threshold]
        print(f"Episodes with max_risk < {threshold}:")
        
    if filtered_eps.empty:
        print("No episodes found matching the criteria.")
    else:
        print(filtered_eps['episode_num'].tolist())

def show_risk_statistics(csv_path, columns):
    """
    Prints descriptive statistics for specified columns of the risk data.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file at {csv_path} was not found.")
        return

    invalid_columns = [col for col in columns if col not in df.columns]
    if invalid_columns:
        print(f"Error: The following columns were not found: {invalid_columns}")
        print("\nAvailable columns are:")
        print(df.columns.tolist())
        return
    
    print(f"Descriptive statistics for columns: {columns}")
    print(df[columns].describe())

def main_risk_logic(args):
    """
    Main logic for orchestrating risk analysis tasks based on parsed arguments.
    """
    # --- Determine directories based on --dataset or specific args ---
    input_dir = args.input_directory
    output_dir = args.output_directory
    csv_dir = args.risk_csv_dir

    if args.dataset:
        dataset_name = args.dataset
        if not input_dir:
            input_dir = f"data/graphical_final/{dataset_name}"
        if not output_dir:
            output_dir = f"data/frame_targets/{dataset_name}"
        if not csv_dir:
            csv_dir = f"data/frame_targets/{dataset_name}/risk_data.csv"

    # --- Execute requested operation ---
    if args.run_analysis:
        if not all([input_dir, output_dir, args.num_episodes is not None]):
            print("Error: --run_analysis requires --num_episodes and either --dataset or both --input_directory and --output_directory.")
            return
        run_risk_analysis(input_dir, output_dir, args.num_episodes)
    
    elif args.run_on_all_episodes:
        if not all([input_dir, output_dir, args.output_filename]):
            print("Error: --run_on_all_episodes requires --output_filename and either --dataset or both --input_directory and --output_directory.")
            return
        run_risk_analysis_on_all_episodes(input_dir, output_dir, args.output_filename)

    elif args.extract_large_risk_eps:
        if not all([csv_dir, args.threshold is not None]):
            print("Error: --extract_large_risk_eps requires --threshold and either --dataset or --risk_csv_dir.")
            return
        extract_risk_episodes(csv_dir, args.threshold, high_risk=True)

    elif args.extract_low_risk_eps:
        if not all([csv_dir, args.threshold is not None]):
            print("Error: --extract_low_risk_eps requires --threshold and either --dataset or --risk_csv_dir.")
            return
        extract_risk_episodes(csv_dir, args.threshold, high_risk=False)

    elif args.risk_statistics:
        if not all([csv_dir, args.columns]):
            print("Error: --risk_statistics requires --columns and either --dataset or --risk_csv_dir.")
            return
        show_risk_statistics(csv_dir, args.columns)
    
    else:
        print("No action specified. Please use one of the flags: --run_analysis, --extract_large_risk_eps, --extract_low_risk_eps, --risk_statistics.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run or analyze risk analysis data.")
    
    # --- Operation Flags ---
    parser.add_argument("--run_analysis", action="store_true", help="Run the full risk analysis and generate a CSV.")
    parser.add_argument("--run_on_all_episodes", action="store_true", help="Run risk analysis on all episodes in a directory.")
    parser.add_argument("--extract_large_risk_eps", action="store_true", help="Extract episodes with risk above a threshold.")
    parser.add_argument("--extract_low_risk_eps", action="store_true", help="Extract episodes with risk below a threshold.")
    parser.add_argument("--risk_statistics", action="store_true", help="Show statistics for specified columns.")

    # --- Path & Data Arguments ---
    parser.add_argument("--dataset", type=str, help="Dataset name to use default directory structures (e.g., 'L2D', 'NUP').")
    parser.add_argument("--input_directory", type=str, help="Override default input directory for graph JSON files.")
    parser.add_argument("--output_directory", type=str, help="Override default output directory for the CSV file.")
    parser.add_argument("--risk_csv_dir", type=str, help="Override default path to the risk_data.csv file for analysis.")
    parser.add_argument("--output_filename", type=str, help="Output filename for the JSON file (e.g., 'risk_results.json').")

    # --- Parameter Arguments ---
    parser.add_argument("--num_episodes", type=int, help="Number of episodes to process (for --run_analysis).")
    parser.add_argument("--threshold", type=float, help="Risk threshold for episode extraction.")
    parser.add_argument("--columns", type=str, nargs='+', help="List of columns for statistics.")

    args = parser.parse_args()
    main_risk_logic(args)
