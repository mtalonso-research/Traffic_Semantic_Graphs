import os
import json
import glob
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import json
import glob
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


class EpisodeDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        labels_json_path: str,
        transform=None,
        episode_dir_format: str = "Episode{episode_id:06d}",
    ):
        """
        Args:
            root_dir: directory containing episode subfolders.
            labels_json_path: JSON mapping {episode_id_str: [label_0, ..., label_K-1]}.
            transform: torchvision-style transform applied to each frame.
            episode_dir_format: format string to map int episode_id -> subdir name.
                E.g. "Episode{episode_id:06d}" or "Episode{episode_id:06d}/observation.images.front_left".
        """
        self.root_dir = root_dir
        self.transform = transform
        self.episode_dir_format = episode_dir_format

        with open(labels_json_path, "r") as f:
            raw_labels = json.load(f)

        # sort keys numerically for reproducibility
        self.episode_ids: List[str] = sorted(raw_labels.keys(), key=lambda x: int(x))
        self.samples = []

        skipped_missing_dir = 0
        skipped_no_frames = 0

        for eid in self.episode_ids:
            int_id = int(eid)

            # Map episode id -> folder pattern
            episode_rel = episode_dir_format.format(episode_id=int_id)
            episode_dir = os.path.join(root_dir, episode_rel)

            if not os.path.isdir(episode_dir):
                print(
                    f"[EpisodeDataset] WARNING: episode dir not found, skipping: "
                    f"{episode_dir} (episode id {eid}, mapped '{episode_rel}')"
                )
                skipped_missing_dir += 1
                continue

            frame_paths = sorted(glob.glob(os.path.join(episode_dir, "*.jpg")))
            if len(frame_paths) == 0:
                print(
                    f"[EpisodeDataset] WARNING: no .jpg frames in {episode_dir}, skipping episode {eid}"
                )
                skipped_no_frames += 1
                continue

            target = torch.tensor(raw_labels[eid], dtype=torch.long)

            self.samples.append(
                {
                    "episode_id": eid,
                    "frame_paths": frame_paths,
                    "target": target,
                }
            )

        if len(self.samples) == 0:
            raise RuntimeError(
                "EpisodeDataset found no valid episodes with frames. "
                f"Skipped {skipped_missing_dir} with missing dirs and "
                f"{skipped_no_frames} with no frames. "
                "Check your root_dir / episode_dir_format / labels_json_path."
            )

        print(
            f"[EpisodeDataset] Loaded {len(self.samples)} episodes "
            f"(skipped {skipped_missing_dir} missing-dir, "
            f"{skipped_no_frames} no-frames)."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        frame_paths = sample["frame_paths"]
        target = sample["target"]
        episode_id = sample["episode_id"]

        frames = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            if self.transform is None:
                raise RuntimeError("EpisodeDataset requires a transform (e.g. Resize+ToTensor+Normalize).")
            img = self.transform(img)
            frames.append(img)

        frames_tensor = torch.stack(frames, dim=0)  # (T, C, H, W)
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