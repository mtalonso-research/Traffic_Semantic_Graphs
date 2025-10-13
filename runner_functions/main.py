import json
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
from torch_geometric.utils import to_dense_batch
import torch.optim as optim
import json

from functions.load_data_L2D import data_downloader
from functions.process_tabular_data_L2D import process_tabular_data
from functions.process_tags_L2D import add_data_tags
from functions.process_frames_L2D import process_frames
from functions.process_lanes_L2D import lane_processing
from functions.graphs import generate_graphs

import argparse
parser = argparse.ArgumentParser(description="Process L2D data.")
parser.add_argument("--min_ep", type=int, default=0, help="Minimum episode number to process.")
parser.add_argument("--max_ep", type=int, default=-1, help="Maximum episode number to process. Use -1 to process all available episodes.")
args = parser.parse_args()

def default_l2d_processing(min_ep,max_ep=-1):
    print('========== Downloading Data ==========')
    data_downloader(min_ep,max_ep,n_secs=3,
                    features={"tabular": True,
                            "frames": {
                                    'observation.images.front_left': True,
                                    'observation.images.left_backward': False,
                                    'observation.images.left_forward': False,
                                    'observation.images.map': False,
                                    'observation.images.rear': False,
                                    'observation.images.right_backward': False,
                                    'observation.images.right_forward': False,
                                }},
                    tabular_data_dir = './data/raw/L2D/tabular',
                    frames_dir = './data/raw/L2D/frames',
                    metadata_dir = './data/raw/L2D/metadata',
                    )
    print("========== Processing Tabular Data ==========")
    process_tabular_data(min_ep,max_ep,
                        overwrite=True, process_columns=True, 
                        process_osm=True, process_turning=True,
                        time_sleep=2,
                        source_dir = './data/raw/L2D/tabular',
                        output_dir_processed = './data/processed_data/L2D',
                        output_dir_tags = './data/semantic_tags/L2D')
    print("========== Adding Tags ==========")
    add_data_tags(min_ep,max_ep,
                    data_dir = './data/processed/L2D',
                    tags_dir='./data/semantic_tags/L2D')
    print("========== Processing Frames ==========")
    process_frames(min_ep,max_ep,
                cameras_on=["observation.images.front_left"],
                run_dict={"detection": True,
                            "depth": True,
                            "speed": True,
                            'overwrite': True},
                    verbose=False,
                    input_base_dir = './data/raw/L2D/frames',
                    output_base_dir = './data/processed_frames/L2D')
    _ = lane_processing(min_ep,max_ep,
                        output_base_dir = './data/processed_frames/L2D')
    print("========== Generate Graphs ==========")
    generate_graphs(min_ep,max_ep,
                    source_data_dir = './data/processed/L2D',
                    processed_frame_dir = './data/processed_frames/L2D',
                    output_dir = './data/graphical/L2D')

if __name__ == "__main__":
    min_episode = args.min_ep
    max_episode = args.max_ep
    default_l2d_processing(min_episode, max_episode)