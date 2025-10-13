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
sys.path.append(os.path.abspath(".."))

from functions.load_data_L2D import data_downloader
from functions.process_tabular_data_L2D import process_tabular_data
from functions.process_tags_L2D import add_data_tags
from functions.process_frames_L2D import process_frames
from functions.process_lanes_L2D import lane_processing
from functions.graphs import generate_graphs

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
                                }
                            })
    print("========== Processing Tabular Data ==========")
    process_tabular_data(min_ep,max_ep,
                        overwrite=True, process_columns=True, 
                        process_osm=True, process_turning=True,
                        time_sleep=2)
    print("========== Adding Tags ==========")
    add_data_tags(min_ep,max_ep)
    print("========== Processing Frames ==========")
    process_frames(min_ep,max_ep,
                cameras_on=["observation.images.front_left"],
                run_dict={"detection": True,
                            "depth": True,
                            "speed": True,
                            'overwrite': True},
                    verbose=False)
    _ = lane_processing(min_ep,max_ep)
    print("========== Generate Graphs ==========")
    generate_graphs(min_ep,max_ep)