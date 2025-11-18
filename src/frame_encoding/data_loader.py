import os
import json
import glob
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class EpisodeDataset(Dataset):
    """
    One sample = one episode.
    Returns:
        frames:  (T, C, H, W) float tensor
        target:  (K,) long tensor
        episode_id: str
    """
    def __init__(
        self,
        root_dir: str,
        labels_json_path: str,
        transform=None,
        episode_dir_format: str = "Episode{episode_id:06d}/observation.images.front_left",
    ):
        """
        Args:
            root_dir: directory containing Episode*/ subfolders.
            labels_json_path: path to JSON mapping {episode_id_str: [label_0, ..., label_K-1]}.
            transform: torchvision-style transform applied to each frame (PIL -> tensor).
            episode_dir_format: how to map an integer episode_id to folder name.
                Default: Episode000123 for id "123".
                If your folders are named differently (e.g. "Episode27441"),
                change this to "Episode{episode_id}".
        """
        self.root_dir = root_dir
        self.transform = transform
        self.episode_dir_format = episode_dir_format

        # Load label dict
        with open(labels_json_path, "r") as f:
            raw_labels = json.load(f)

        # Sort episode IDs numerically for reproducibility
        # Keys are strings; convert to int for sorting
        self.episode_ids: List[str] = sorted(raw_labels.keys(), key=lambda x: int(x))

        # Build an internal list of (episode_id, frame_paths, label_tensor)
        self.samples = []
        for eid in self.episode_ids:
            int_id = int(eid)

            # Map JSON episode ID -> subdirectory name
            episode_dir_name = episode_dir_format.format(episode_id=int_id)
            episode_dir = os.path.join(root_dir, episode_dir_name)

            if not os.path.isdir(episode_dir):
                raise FileNotFoundError(
                    f"Episode directory not found: {episode_dir} "
                    f"(episode id {eid}, mapped name '{episode_dir_name}')"
                )

            # Collect all jpg frames for this episode
            frame_paths = sorted(
                glob.glob(os.path.join(episode_dir, "*.jpg"))
            )

            if len(frame_paths) == 0:
                raise RuntimeError(f"No .jpg frames found in {episode_dir}")

            # Convert label list to tensor
            label_list = raw_labels[eid]
            target = torch.tensor(label_list, dtype=torch.long)

            self.samples.append(
                {
                    "episode_id": eid,
                    "frame_paths": frame_paths,
                    "target": target,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[idx]
        frame_paths = sample["frame_paths"]
        target = sample["target"]
        episode_id = sample["episode_id"]

        frames = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            else:
                # If no transform is provided, convert to a simple tensor [0,1]
                img = torch.from_numpy(
                    (np.array(img).astype("float32") / 255.0).transpose(2, 0, 1)
                )
            frames.append(img)

        # Stack into (T, C, H, W)
        frames_tensor = torch.stack(frames, dim=0)

        return frames_tensor, target, episode_id
    
from typing import List, Tuple

def collate_episodes(batch: List[Tuple[torch.Tensor, torch.Tensor, str]]):
    """
    batch: list of (frames, target, episode_id)
        frames: (T_i, C, H, W)
        target: (K,)
        episode_id: str

    returns:
        frames_list: list of length B, each (T_i, C, H, W)
        targets: (B, K)
        episode_ids: list[str]
    """
    frames_list, targets_list, ids_list = zip(*batch)

    # frames_list is already a tuple of variable-length tensors; keep as list
    frames_list = list(frames_list)
    targets = torch.stack(targets_list, dim=0)  # (B, K)
    episode_ids = list(ids_list)

    return frames_list, targets, episode_ids