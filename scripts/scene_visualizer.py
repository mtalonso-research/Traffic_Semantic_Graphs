import os
import argparse

from src.visualizations import scene_visualizer

parser = argparse.ArgumentParser(description="Visualize data.")

parser.add_argument("--dataset_path", type=str, default='./data/graphical_final/nuplan_boston')
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--frame", type=lambda x: None if x.lower() == "none" else int(x), default=None)

args = parser.parse_args()

def visualizations(dataset_path, episode, frame):
    scene_visualizer(dataset_path,episode,frame)

if __name__ == "__main__":
    visualizations(dataset_path=args.dataset_path,
                   episode=args.episode,
                   frame=args.frame)